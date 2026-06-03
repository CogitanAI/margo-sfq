"""
graph.py — topology-faithful, lossless in-memory model of a JoSIM/SPICE netlist.

This is the netlist representation this toolkit edits. Design goals:

1. **Lossless round-trip.** Every element carries its *raw* value token(s) exactly
   as parsed. An unmodified element re-emits verbatim, so parse->emit is byte-faithful
   for the parts nobody touched. Only when a generator *changes* a value do we fall
   back to ``units.format_value``.
2. **Numeric access for the generator.** Each element also exposes parsed floats
   (SI base units) so search/DRC/learning can reason about inductance, Ic, etc.
3. **Real connectivity.** Elements connect to *named nets* (the SPICE node labels),
   so a circuit graph can be reconstructed directly.

We model the SPICE element types that appear in RSFQ netlists: B (JJ), L
(inductor, with K mutual coupling), R, C, I/V sources, T (transmission line / PTL),
and X (subckt instance). Directives we must preserve structurally: ``.model``,
``.param``, ``.subckt``/``.ends``; everything else (``.tran``, ``.print``, ``.end``,
comments) rides along as opaque control lines so emit is faithful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import units


# --------------------------------------------------------------------------- #
# Element types
# --------------------------------------------------------------------------- #
#
# Every element subclass stores:
#   name      : the SPICE designator (e.g. "B01", "L3", "X0")
#   nodes     : ordered list of net names it attaches to
#   raw        : the original token string for its primary value (None if N/A)
#   value     : parsed float in SI base units (None if not a plain numeric value,
#                e.g. a pwl(...) source function, or a JJ whose "value" is a model)
#   modified  : set True by a generator when it changes `value`; controls whether
#                emit uses `raw` (verbatim) or re-formats from `value`.
#   params    : trailing key=value params (area=2.16, Z0=5, td=10p, ...), values
#                kept as raw strings to preserve fidelity; parse on demand.
#
# We keep `params` as an ordered dict of raw strings. A helper resolves them to
# floats when the generator needs to reason about them.


@dataclass
class Element:
    name: str
    nodes: list[str]
    raw: Optional[str] = None
    value: Optional[float] = None
    modified: bool = False
    params: dict[str, str] = field(default_factory=dict)

    # SPICE letter prefix this element emits with. Subclasses set it.
    letter: str = ""

    def param_float(self, key: str) -> Optional[float]:
        """Parse a trailing param (e.g. area=2.16) to float, or None if absent."""
        if key not in self.params:
            return None
        tok = self.params[key]
        try:
            return units.parse_value(tok)
        except ValueError:
            return None

    def set_value(self, new_value: float) -> None:
        """Generator entry point: change the numeric value and mark dirty."""
        self.value = new_value
        self.modified = True


@dataclass
class JJ(Element):
    """Josephson junction:  Bxxx n+ n- [phasenode] model [area=... ...].

    The "value" of a JJ is its model name, not a number. Area (or ic) lives in
    params. JoSIM allows an optional third (phase) node before the model name;
    we keep the raw node list so we can re-emit it exactly.
    """
    letter: str = "B"
    model: str = ""
    # Some JoSIM JJs list an explicit phase node as a 3rd node; tracked via nodes.


@dataclass
class Inductor(Element):
    letter: str = "L"


@dataclass
class Resistor(Element):
    letter: str = "R"


@dataclass
class Capacitor(Element):
    letter: str = "C"


@dataclass
class Source(Element):
    """Current (I) or voltage (V) source. `func` holds a non-numeric drive such
    as ``pwl(0 0 5p 280u)`` or ``pulse(...)``; when set, `value`/`raw` are None
    and we emit `func` verbatim."""
    letter: str = "I"
    func: Optional[str] = None


@dataclass
class TLine(Element):
    """Lossless transmission line / PTL:  Txxx n1 g1 n2 g2 Z0=.. td=..  (4 nodes)."""
    letter: str = "T"


@dataclass
class Mutual(Element):
    """Mutual inductive coupling:  Kxxx Lname1 Lname2 k.

    Note `nodes` here actually holds *inductor names*, not net names — K couples
    two inductors. `value` is the coupling coefficient k (unitless, 0..1)."""
    letter: str = "K"


@dataclass
class SubcktInstance(Element):
    """Hierarchical instance:  Xxxx ...  JoSIM accepts two orderings:
        Xname node node ... subcktname      (nodes first)
        Xname subcktname node node ...      (subckt first)
    We record which with `subckt_first` so emit matches the source."""
    letter: str = "X"
    subckt: str = ""
    subckt_first: bool = False


# --------------------------------------------------------------------------- #
# Directives we must keep structured
# --------------------------------------------------------------------------- #


@dataclass
class Model:
    """.model NAME jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)

    Emitted verbatim from `raw` unless a sweep/generator edits a param (sets
    `modified`), in which case the line is rebuilt from name/mtype/params."""
    name: str
    mtype: str                       # "jj", "res", ...
    params: dict[str, str] = field(default_factory=dict)
    raw: str = ""                    # full original line (verbatim re-emit)
    modified: bool = False

    def set_param(self, key: str, value: str) -> None:
        self.params[key.lower()] = value
        self.modified = True

    def rebuild(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f".model {self.name} {self.mtype}({inner})"

    def text(self) -> str:
        return self.rebuild() if self.modified else self.raw


@dataclass
class Param:
    """.param name=expr  — value kept as a raw expression string (may reference
    other params)."""
    name: str
    expr: str
    raw: str = ""


@dataclass
class RawLine:
    """An opaque source line we preserve verbatim: comments (*...), blanks, and
    control directives we don't model structurally (.tran, .print, .iv, .plot,
    .end, .options, ...). Carrying these keeps parse->emit byte-faithful."""
    text: str


@dataclass
class Subckt:
    """.subckt NAME port port ...  ... .ends — a reusable cell definition.

    The body is parsed into its own NetlistGraph (own local models/params scope).
    `raw_header` / `raw_footer` preserve the exact .subckt / .ends lines."""
    name: str
    ports: list[str]
    graph: "NetlistGraph"
    raw_header: str = ""
    raw_footer: str = ".ends"


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #

# An item in a netlist scope, in source order. Union of the structured types
# plus RawLine for passthrough. Subckt blocks nest their own NetlistGraph.
Item = object  # one of: Element subclass | Model | Param | Subckt | RawLine


@dataclass
class NetlistGraph:
    """A netlist scope: the top-level deck, or one subckt body.

    `items` is the ordered source stream — iterate it to emit faithfully. The
    name-indexed dicts (`_elements`, `models`, `params`, `subckts`) point at the
    *same* objects for O(1) lookup by the generator/DRC. Mutating an element in
    place (via set_value) is reflected on emit automatically.
    """
    items: list = field(default_factory=list)
    models: dict[str, Model] = field(default_factory=dict)
    params: dict[str, Param] = field(default_factory=dict)
    subckts: dict[str, Subckt] = field(default_factory=dict)

    # ---- mutation helpers used by the parser as it builds the scope ----

    def add(self, item) -> None:
        self.items.append(item)
        if isinstance(item, Model):
            self.models[item.name.lower()] = item
        elif isinstance(item, Param):
            self.params[item.name.lower()] = item
        elif isinstance(item, Subckt):
            self.subckts[item.name.lower()] = item

    # ---- convenience accessors for the generator / DRC ----

    @property
    def elements(self) -> list:
        return [it for it in self.items if isinstance(it, Element)]

    def nets(self) -> set[str]:
        """All net names referenced by elements (excludes K's inductor refs)."""
        out: set[str] = set()
        for el in self.elements:
            if isinstance(el, Mutual):
                continue
            out.update(el.nodes)
        return out

    def junctions(self) -> list[JJ]:
        return [e for e in self.elements if isinstance(e, JJ)]

    def inductors(self) -> list[Inductor]:
        return [e for e in self.elements if isinstance(e, Inductor)]

    def by_name(self, name: str) -> Optional[Element]:
        for e in self.elements:
            if e.name.lower() == name.lower():
                return e
        return None
