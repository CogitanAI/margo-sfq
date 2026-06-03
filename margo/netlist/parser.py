"""
parser.py — JoSIM/SPICE ``.cir`` text -> NetlistGraph.

Strategy (multi-pass, scope-aware):

1. **Physical -> logical lines.** Merge ``+`` continuation lines; keep comments
   and blanks as their own logical lines so emit can replay them.
2. **Block split.** Carve out ``.subckt`` ... ``.ends`` blocks (each parsed
   recursively into its own scope, since models/params are scope-local). Subckt
   *names* are collected globally first — JoSIM resolves ``X`` instances against
   a global subckt namespace, and definitions may follow uses.
3. **Per-scope pre-scan.** Collect ``.model`` names declared in this scope before
   classifying elements. RSFQ subckts declare the JJ ``.model`` at the *end* of
   the body, after the ``B`` elements that reference it — so we must know the
   model names before we can split a JJ line into nodes vs. model.
4. **Element classification.** First letter of the designator picks the type.
   For ``B`` (JJ) the model token is the one in the known-model set (resolving
   the optional phase node). For ``X`` the subckt token is the one in the known
   global subckt set (resolving subckt-first vs. nodes-first ordering).

The bar for correctness is JoSIM-output equality after parse->emit, not byte
identity. We preserve raw value tokens
so unmodified elements re-emit stably; whitespace may normalize.
"""
from __future__ import annotations

import re

from . import units
from .graph import (
    Capacitor,
    Inductor,
    JJ,
    Model,
    Mutual,
    NetlistGraph,
    Param,
    RawLine,
    Resistor,
    Source,
    Subckt,
    SubcktInstance,
    TLine,
)


# --------------------------------------------------------------------------- #
# Line preprocessing
# --------------------------------------------------------------------------- #


def _logical_lines(text: str) -> list[str]:
    """Merge ``+`` continuation lines into their predecessor.

    Comment (``*``) and blank lines are preserved as standalone logical lines.
    A continuation line (first non-space char ``+``) appends to the last
    *non-comment* logical line.
    """
    out: list[str] = []
    for phys in text.splitlines():
        stripped = phys.strip()
        if stripped.startswith("+"):
            cont = stripped[1:].strip()
            # Attach to the most recent real (non-comment, non-blank) line.
            for i in range(len(out) - 1, -1, -1):
                s = out[i].strip()
                if s and not s.startswith("*"):
                    out[i] = out[i].rstrip() + " " + cont
                    break
            else:
                out.append(phys)  # stray '+', keep as-is
        else:
            out.append(phys)
    return out


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("*")


# --------------------------------------------------------------------------- #
# Subckt name harvesting (global) + block carving
# --------------------------------------------------------------------------- #


def _collect_subckt_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        s = line.strip()
        if s.lower().startswith(".subckt"):
            toks = s.split()
            if len(toks) >= 2:
                names.add(toks[1].lower())
    return names


# --------------------------------------------------------------------------- #
# Model param parsing
# --------------------------------------------------------------------------- #

_MODEL_RE = re.compile(
    r"^\.model\s+(\S+)\s+(\w+)\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL
)


def _parse_model(line: str) -> Model:
    m = _MODEL_RE.match(line.strip())
    if not m:
        # Fall back: keep name/type best-effort, preserve raw for emit.
        toks = line.strip().split()
        name = toks[1] if len(toks) > 1 else ""
        return Model(name=name, mtype="", params={}, raw=line.rstrip())
    name, mtype, inner = m.group(1), m.group(2), m.group(3)
    params: dict[str, str] = {}
    for kv in re.split(r"[,\s]+", inner.strip()):
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        params[k.strip().lower()] = v.strip()
    return Model(name=name, mtype=mtype.lower(), params=params, raw=line.rstrip())


def _collect_model_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        s = line.strip()
        if s.lower().startswith(".model"):
            toks = s.split()
            if len(toks) >= 2:
                names.add(toks[1].lower())
    return names


# --------------------------------------------------------------------------- #
# Element parsing helpers
# --------------------------------------------------------------------------- #


def _split_params(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    """Partition trailing key=value params from leading positional tokens."""
    positional: list[str] = []
    params: dict[str, str] = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            params[k.strip().lower()] = v.strip()
        else:
            positional.append(t)
    return positional, params


def _parse_jj(name: str, rest: list[str], model_names: set[str]) -> JJ:
    positional, params = _split_params(rest)
    # The model is the positional token in the known-model set; everything before
    # it is nodes. This resolves the optional 3rd (phase) node automatically.
    model_idx = None
    for i, t in enumerate(positional):
        if t.lower() in model_names:
            model_idx = i
            break
    if model_idx is None:
        # No known model matched (e.g. model declared elsewhere / typo). Assume
        # the last non-numeric positional token is the model.
        for i in range(len(positional) - 1, -1, -1):
            if not units.is_value(positional[i]):
                model_idx = i
                break
        if model_idx is None:
            model_idx = len(positional) - 1
    nodes = positional[:model_idx]
    model = positional[model_idx] if model_idx < len(positional) else ""
    return JJ(name=name, nodes=nodes, model=model, params=params, letter="B")


def _parse_two_node_value(cls, letter: str, name: str, rest: list[str]):
    """L/R/C: 2 nodes, then a scalar value, then optional key=value params."""
    positional, params = _split_params(rest)
    nodes = positional[:2]
    raw = positional[2] if len(positional) > 2 else None
    value = None
    if raw is not None:
        try:
            value = units.parse_value(raw)
        except ValueError:
            value = None
    return cls(name=name, nodes=nodes, raw=raw, value=value,
               params=params, letter=letter)


def _parse_source(name: str, letter: str, rest: list[str]) -> Source:
    """I/V: 2 nodes, then either a scalar value or a drive function (pwl/pulse/
    sin/...). The function is captured verbatim (joined) and emitted as-is."""
    nodes = rest[:2]
    tail = rest[2:]
    if not tail:
        return Source(name=name, nodes=nodes, letter=letter)
    joined = " ".join(tail)
    # Numeric scalar source?
    if len(tail) == 1 and units.is_value(tail[0]):
        return Source(name=name, nodes=nodes, raw=tail[0],
                      value=units.parse_value(tail[0]), letter=letter)
    return Source(name=name, nodes=nodes, func=joined, letter=letter)


def _parse_tline(name: str, rest: list[str]) -> TLine:
    """T: 4 nodes then Z0=/td= params."""
    positional, params = _split_params(rest)
    nodes = positional[:4]
    return TLine(name=name, nodes=nodes, params=params, letter="T")


def _parse_mutual(name: str, rest: list[str]) -> Mutual:
    """K Lname1 Lname2 coeff — nodes here are inductor *names*."""
    ind_names = rest[:2]
    coeff_raw = rest[2] if len(rest) > 2 else None
    value = None
    if coeff_raw is not None:
        try:
            value = units.parse_value(coeff_raw)
        except ValueError:
            value = None
    return Mutual(name=name, nodes=ind_names, raw=coeff_raw, value=value,
                  letter="K")


def _parse_subckt_instance(name: str, rest: list[str],
                           subckt_names: set[str]) -> SubcktInstance:
    """X: subckt token is the one in the global subckt set; its position decides
    subckt-first vs nodes-first ordering."""
    positional, params = _split_params(rest)
    sub_idx = None
    for i, t in enumerate(positional):
        if t.lower() in subckt_names:
            sub_idx = i
            break
    if sub_idx is None:
        # Unknown subckt (e.g. external lib). JoSIM convention: subckt name last.
        sub_idx = len(positional) - 1
    subckt = positional[sub_idx] if 0 <= sub_idx < len(positional) else ""
    if sub_idx == 0:
        nodes = positional[1:]
        subckt_first = True
    else:
        nodes = positional[:sub_idx]
        subckt_first = False
    return SubcktInstance(name=name, nodes=nodes, subckt=subckt,
                          subckt_first=subckt_first, params=params, letter="X")


_ELEM_DISPATCH = {
    "L": (Inductor, "L"),
    "R": (Resistor, "R"),
    "C": (Capacitor, "C"),
}


def _parse_element(line: str, model_names: set[str], subckt_names: set[str]):
    toks = line.split()
    name = toks[0]
    rest = toks[1:]
    letter = name[0].upper()
    if letter == "B":
        return _parse_jj(name, rest, model_names)
    if letter in _ELEM_DISPATCH:
        cls, lt = _ELEM_DISPATCH[letter]
        return _parse_two_node_value(cls, lt, name, rest)
    if letter in ("I", "V"):
        return _parse_source(name, letter, rest)
    if letter == "T":
        return _parse_tline(name, rest)
    if letter == "K":
        return _parse_mutual(name, rest)
    if letter == "X":
        return _parse_subckt_instance(name, rest, subckt_names)
    # Unknown element type (E/F/G/H controlled sources, P, ...): preserve verbatim.
    return RawLine(text=line.rstrip())


# --------------------------------------------------------------------------- #
# Scope parsing
# --------------------------------------------------------------------------- #


def _parse_scope(lines: list[str], subckt_names: set[str],
                 model_names: set[str]) -> NetlistGraph:
    """Parse a flat list of logical lines (no enclosing .subckt) into a graph.

    Nested .subckt blocks are carved out and recursed. `subckt_names` and
    `model_names` are the global sets used to disambiguate X instances and JJ
    model tokens — JoSIM resolves both against a global namespace, and some
    RSFQ netlists declare JJ models at top level yet reference them inside
    subckts.
    """
    g = NetlistGraph()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if _is_comment_or_blank(line):
            g.add(RawLine(text=line.rstrip("\n")))
            i += 1
            continue

        low = s.lower()
        if low.startswith(".subckt"):
            # Carve to matching .ends (handle nesting defensively).
            header = line
            toks = s.split()
            sub_name = toks[1] if len(toks) > 1 else ""
            ports = toks[2:]
            depth = 1
            body: list[str] = []
            j = i + 1
            footer = ".ends"
            while j < n:
                js = lines[j].strip().lower()
                if js.startswith(".subckt"):
                    depth += 1
                elif js.startswith(".ends") or js == ".end":
                    depth -= 1
                    if depth == 0:
                        footer = lines[j].rstrip("\n")
                        break
                body.append(lines[j])
                j += 1
            sub_graph = _parse_scope(body, subckt_names, model_names)
            sub = Subckt(name=sub_name, ports=ports, graph=sub_graph,
                         raw_header=header.rstrip("\n"), raw_footer=footer)
            g.add(sub)
            i = j + 1
            continue

        if low.startswith(".model"):
            g.add(_parse_model(line))
            i += 1
            continue

        if low.startswith(".param"):
            body = s[len(".param"):].strip()
            if "=" in body:
                pname, expr = body.split("=", 1)
                g.add(Param(name=pname.strip(), expr=expr.strip(),
                            raw=line.rstrip("\n")))
            else:
                g.add(RawLine(text=line.rstrip("\n")))
            i += 1
            continue

        if low.startswith("."):
            # Any other directive (.tran/.print/.iv/.plot/.end/.options/...).
            g.add(RawLine(text=line.rstrip("\n")))
            i += 1
            continue

        # Otherwise: a circuit element.
        g.add(_parse_element(line, model_names, subckt_names))
        i += 1

    return g


def parse_cir(text: str) -> NetlistGraph:
    """Parse a full JoSIM ``.cir`` deck into a NetlistGraph."""
    lines = _logical_lines(text)
    subckt_names = _collect_subckt_names(lines)
    model_names = _collect_model_names(lines)
    return _parse_scope(lines, subckt_names, model_names)
