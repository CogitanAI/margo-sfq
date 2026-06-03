"""
drc.py — fast, pre-JoSIM validity gate in netlist space.

A cheap structural/physics screen run on every candidate before spending a ~100 ms
JoSIM call. The rules, expressed directly on a ``NetlistGraph``:

  * floating nets   — every non-ground net must touch >= 2 element terminals
                      (a 1-terminal net is a dangling wire / broken loop).
  * bias present    — an active cell needs >= 1 bias current source.
  * JJ shunting     — each JJ needs a damping resistor on one of its nodes
                      (RSFQ JJs are externally shunted; an unshunted JJ is
                      underdamped and latches). Heuristic, flagged as such.
  * param bounds    — per-JJ Ic in a sane window; L > 0; R >= 0.
  * betaL band      — each inductive storage loop containing a JJ should have
                      betaL = 2*pi*L*Ic/Phi0 in [0.5, 2.0]; outside that the loop
                      can't reliably store a flux quantum.

Checks that need a value we can't resolve (symbolic ``area``/param expressions,
no numeric inductance) are reported as ``unresolved`` rather than guessed — the
gate stays honest. ``check(graph)`` returns a report; ``is_valid(graph)`` is the
boolean accept gate (no error-severity violations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .graph import JJ, Inductor, Mutual, NetlistGraph, Resistor, Source
from . import units


PHI0_WB = 2.0678e-15          # flux quantum (Wb)
BETAL_LO, BETAL_HI = 0.5, 2.0
IC_MIN_UA, IC_MAX_UA = 40.0, 450.0   # sane per-JJ critical current window
GROUND_NETS = {"0", "gnd", "0.0"}


@dataclass
class Violation:
    rule: str
    severity: str                 # "error" | "warning" | "unresolved"
    message: str
    where: Optional[str] = None   # element/net/loop identifier


@dataclass
class DRCReport:
    violations: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, *a, **k) -> None:
        self.violations.append(Violation(*a, **k))


# --------------------------------------------------------------------------- #
# Ic resolution
# --------------------------------------------------------------------------- #


def _jj_ic_ua(jj: JJ, graph: NetlistGraph) -> Optional[float]:
    """Critical current (uA) = model icrit * area, when both are numeric."""
    model = graph.models.get(jj.model.lower())
    if model is None or "icrit" not in model.params:
        return None
    try:
        icrit_a = units.parse_value(model.params["icrit"])
    except ValueError:
        return None
    area = jj.params.get("area")
    if area is None:
        area_v = 1.0
    else:
        try:
            area_v = units.parse_value(area)
        except ValueError:
            return None        # symbolic area — unresolved
    return icrit_a * area_v * 1e6


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _check_floating_nets(g: NetlistGraph, rep: DRCReport) -> None:
    degree: dict[str, int] = {}
    for el in g.elements:
        if isinstance(el, Mutual):
            continue          # K references inductor names, not nets
        # A JJ's optional 3rd node is a phase-tracking node, not an electrical
        # net — it legitimately attaches to that one JJ. Only count terminals.
        nodes = el.nodes[:2] if isinstance(el, JJ) else el.nodes
        for n in nodes:
            degree[n] = degree.get(n, 0) + 1
    # Ports of an enclosing subckt connect outward, so a net that is also a port
    # name is legitimately single-terminal here. Callers pass ports via g? We
    # don't have them at top level; treat ground and 1-degree-only conservatively.
    for net, deg in degree.items():
        if net.lower() in GROUND_NETS:
            continue
        if deg < 2:
            rep.add("floating_net", "error",
                    f"net {net!r} touches only {deg} terminal(s)", where=net)


def _check_bias_present(g: NetlistGraph, rep: DRCReport) -> None:
    has_bias = any(
        isinstance(el, Source) and el.letter == "I"
        for el in _all_elements(g)
    )
    if not has_bias:
        rep.add("bias_present", "warning",
                "no current source found — cell may have no bias")


def _check_jj_shunt(g: NetlistGraph, rep: DRCReport) -> None:
    # Map net -> resistors touching it.
    res_nets: set[str] = set()
    for el in g.elements:
        if isinstance(el, Resistor):
            res_nets.update(el.nodes)
    for jj in g.junctions():
        if not any(n in res_nets for n in jj.nodes):
            rep.add("jj_shunt", "warning",
                    f"JJ {jj.name} has no resistor on either node "
                    f"(possibly unshunted/underdamped)", where=jj.name)


def _check_param_bounds(g: NetlistGraph, rep: DRCReport) -> None:
    for jj in g.junctions():
        ic = _jj_ic_ua(jj, g)
        if ic is None:
            rep.add("jj_ic_bounds", "unresolved",
                    f"JJ {jj.name}: Ic not numerically resolvable", where=jj.name)
        elif not (IC_MIN_UA <= ic <= IC_MAX_UA):
            rep.add("jj_ic_bounds", "error",
                    f"JJ {jj.name}: Ic={ic:.1f}uA outside "
                    f"[{IC_MIN_UA},{IC_MAX_UA}]uA", where=jj.name)
    for ind in g.inductors():
        if ind.value is not None and ind.value <= 0:
            rep.add("inductance_bounds", "error",
                    f"inductor {ind.name}: L={ind.value} <= 0", where=ind.name)
    for el in g.elements:
        if isinstance(el, Resistor) and el.value is not None and el.value < 0:
            rep.add("resistance_bounds", "error",
                    f"resistor {el.name}: R={el.value} < 0", where=el.name)


# --------------------------------------------------------------------------- #
# betaL on inductive loops
# --------------------------------------------------------------------------- #


def _fundamental_cycles(edges: list) -> list[list]:
    """Return a fundamental cycle basis of a graph whose edges are circuit
    elements (each contributing its first two `nodes`). Each cycle is the list
    of element objects forming it.

    RSFQ storage loops close *through a junction* (L--JJ loop), so the caller
    passes inductors AND junctions as edges — a pure-inductor graph would miss
    every real storage loop."""
    parent: dict[str, str] = {}

    cycles: list[list] = []
    adj: dict[str, list[tuple[str, object]]] = {}
    for el in edges:
        if len(el.nodes) < 2:
            continue
        a, b = el.nodes[0], el.nodes[1]
        if a == b:
            continue
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        adj.setdefault(a, []).append((b, el))
        adj.setdefault(b, []).append((a, el))

    # Union-find spanning forest; non-tree edges close fundamental cycles.
    visited: set[str] = set()
    for root in list(parent):
        if root in visited:
            continue
        # BFS to build a tree rooted here.
        stack = [root]
        bfs_parent: dict[str, str] = {root: root}
        bfs_edge: dict[str, Inductor] = {}
        while stack:
            u = stack.pop()
            visited.add(u)
            for v, ind in adj.get(u, []):
                if v not in bfs_parent:
                    bfs_parent[v] = u
                    bfs_edge[v] = ind
                    stack.append(v)
        seen_edges: set[int] = set()
        for v in bfs_edge:
            seen_edges.add(id(bfs_edge[v]))
        # Non-tree edges
        for u in list(bfs_parent):
            for v, ind in adj.get(u, []):
                if id(ind) in seen_edges:
                    continue
                if bfs_parent.get(v) == u or bfs_parent.get(u) == v:
                    continue
                # Close the cycle: path u..lca..v through tree + this edge.
                cyc = [ind]
                pu = _path_to_root(u, bfs_parent, bfs_edge)
                pv = _path_to_root(v, bfs_parent, bfs_edge)
                # Symmetric difference of tree paths = the loop's tree edges.
                su = {id(e): e for e in pu}
                sv = {id(e): e for e in pv}
                for k, e in su.items():
                    if k not in sv:
                        cyc.append(e)
                for k, e in sv.items():
                    if k not in su:
                        cyc.append(e)
                seen_edges.add(id(ind))   # avoid emitting the mirror cycle
                cycles.append(cyc)
    return cycles


def _path_to_root(node: str, parent: dict, edge: dict) -> list:
    out: list = []
    cur = node
    while parent.get(cur, cur) != cur:
        out.append(edge[cur])
        cur = parent[cur]
    return out


def _check_betal(g: NetlistGraph, rep: DRCReport) -> None:
    inds = g.inductors()
    jjs = g.junctions()
    if not inds or not jjs:
        return
    # Edges = inductors + junctions; a real storage loop is an L--JJ cycle.
    cycles = _fundamental_cycles(list(inds) + list(jjs))
    for ci, cyc in enumerate(cycles):
        loop_inds = [e for e in cyc if isinstance(e, Inductor)]
        loop_jjs = [e for e in cyc if isinstance(e, JJ)]
        if not loop_inds or not loop_jjs:
            continue            # not an inductive storage loop
        # betaL~1 governs small flux-*storage* loops; a many-inductor cycle is a
        # transmission path (e.g. a JTL's series inductance), where the rule
        # doesn't apply. Restrict to small quantizing loops to avoid false
        # positives. Advisory only (warning) — never blocks the gate.
        if len(loop_inds) > 2:
            continue
        if any(i.value is None for i in loop_inds):
            continue            # symbolic L — cannot evaluate betaL
        loop_L = sum(i.value for i in loop_inds)
        ics = [_jj_ic_ua(jj, g) for jj in loop_jjs]
        ics = [x for x in ics if x is not None]
        if not ics:
            continue
        ic_a = min(ics) * 1e-6
        betal = 2 * 3.141592653589793 * loop_L * ic_a / PHI0_WB
        if not (BETAL_LO <= betal <= BETAL_HI):
            rep.add("betal_band", "warning",
                    f"loop #{ci} (L={loop_L*1e12:.2f}pH, "
                    f"Ic={ic_a*1e6:.0f}uA): betaL={betal:.2f} outside "
                    f"[{BETAL_LO},{BETAL_HI}]", where=f"loop#{ci}")


# --------------------------------------------------------------------------- #
# Helpers + entry points
# --------------------------------------------------------------------------- #


def _all_elements(g: NetlistGraph) -> list:
    """Elements at this scope plus, for top-level decks with no own elements,
    those inside subckts (so a deck defined purely as instantiated cells still
    sees its biases)."""
    out = list(g.elements)
    for sub in g.subckts.values():
        out.extend(sub.graph.elements)
    return out


def check(graph: NetlistGraph, include_subckts: bool = True) -> DRCReport:
    """Run all DRC checks. By default also screens each subckt definition's body
    (where the real JJs/loops live in a hierarchical RSFQ deck)."""
    rep = DRCReport()
    _check_bias_present(graph, rep)        # deck-wide, counts biases in subckts
    _check_scope(graph, rep)
    if include_subckts:
        for sub in graph.subckts.values():
            _check_scope(sub.graph, rep, scope_label=sub.name, ports=sub.ports)
    return rep


def _check_scope(g: NetlistGraph, rep: DRCReport,
                 scope_label: str = "", ports: Optional[list[str]] = None) -> None:
    # Floating-net check must exempt this scope's ports (they connect outward).
    port_set = {p.lower() for p in (ports or [])}
    before = len(rep.violations)
    _check_floating_nets(g, rep)
    if port_set:
        rep.violations = [
            v for v in rep.violations
            if not (v.rule == "floating_net" and v.where
                    and v.where.lower() in port_set)
        ]
    _check_jj_shunt(g, rep)
    _check_param_bounds(g, rep)
    _check_betal(g, rep)
    if scope_label:
        for v in rep.violations[before:]:
            v.where = f"{scope_label}:{v.where}" if v.where else scope_label


def is_valid(graph: NetlistGraph) -> bool:
    """Boolean accept gate: True iff no error-severity violations."""
    return check(graph).ok
