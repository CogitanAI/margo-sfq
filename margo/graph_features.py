"""
graph_features.py — turn a parsed NetlistGraph into a graph the surrogate can ingest.

Netlists name real SPICE nets, so a circuit graph can be built directly:

    node  = a circuit element (JJ / inductor / resistor / current or voltage source)
    edge  = two elements share a (non-ground) net  -> they are physically connected

Node features: element-type one-hot + a couple of normalized numeric channels
(JJ area, inductance in pH, resistance in ohm, source drive magnitude) + degree.
Ground net "0" is excluded from edge formation: every element touches ground, so
including it would make the graph nearly complete and wash out the signal-path
topology that actually distinguishes a JTL chain from a splitter fan.

Output is plain Python (lists/dicts) so this module has NO torch/PyG dependency; the
trainer converts to tensors. A graph-level feature summary is also provided
for a quick MLP baseline.
"""
from __future__ import annotations

import re
from typing import Optional

from margo.netlist import parse_cir
from margo.netlist.graph import JJ, Inductor, Mutual, NetlistGraph, Resistor, Source
from margo.netlist import units

# Element type channels (order fixed — it defines the one-hot layout).
TYPES = ["JJ", "L", "R", "ISRC", "VSRC", "OTHER"]
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}

GROUND_NETS = {"0", "gnd"}

# Rough normalizers so features land near O(1) for a small model.
AREA_NORM = 3.0       # JJ areas ~1.6..2.8
L_PH_NORM = 5.0       # inductances ~0.1..5 pH
R_OHM_NORM = 6.0      # shunt/load ~2..5 ohm
DRIVE_NORM = 300.0    # source drive ~90..280 uA / ~827 uV (scaled below)

_VALUE_TOK = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?[a-zA-Z]*")


def _elem_type(el) -> str:
    if isinstance(el, JJ):
        return "JJ"
    if isinstance(el, Inductor):
        return "L"
    if isinstance(el, Resistor):
        return "R"
    if isinstance(el, Source):
        return "ISRC" if el.letter == "I" else "VSRC"
    return "OTHER"


def _source_drive(el: Source) -> float:
    """Largest-magnitude value token in a source's drive (e.g. the bias hold level
    in pwl(0 0 50p 150u) -> 150e-6, or an SFQ pulse peak). 0 if unparseable."""
    if el.value is not None:
        return abs(el.value)
    if not el.func:
        return 0.0
    best = 0.0
    for tok in _VALUE_TOK.findall(el.func):
        try:
            v = abs(units.parse_value(tok))
        except (ValueError, Exception):  # noqa: BLE001
            continue
        # ignore bare time tokens like "50p" (1e-10 scale) vs currents (~1e-4)
        best = max(best, v)
    return best


def _node_features(el) -> list[float]:
    t = _elem_type(el)
    onehot = [0.0] * len(TYPES)
    onehot[TYPE_IDX[t]] = 1.0
    area = ind_ph = res_ohm = drive = 0.0
    if t == "JJ":
        a = el.param_float("area")
        area = (a or 0.0) / AREA_NORM
    elif t == "L":
        ind_ph = ((el.value or 0.0) * 1e12) / L_PH_NORM     # H -> pH, normalized
    elif t == "R":
        res_ohm = (el.value or 0.0) / R_OHM_NORM
    elif t in ("ISRC", "VSRC"):
        d = _source_drive(el)
        # currents ~1e-4 A; voltages ~1e-3 V — scale to ~O(1) in micro-units
        drive = (d * 1e6) / DRIVE_NORM
    return onehot + [area, ind_ph, res_ohm, drive]


# total node-feature width (must match _node_features output length)
NODE_DIM = len(TYPES) + 4


def graph_from_netlist(g: NetlistGraph) -> dict:
    """Build the element graph. Returns
    {"x": [[feat...]], "edge_index": [[src...],[dst...]], "node_names": [...]}.
    Edges are undirected (both directions emitted) and exclude the ground net."""
    elems = [e for e in g.elements if not isinstance(e, Mutual)]
    names = [e.name for e in elems]
    idx = {e.name: i for i, e in enumerate(elems)}
    x = [_node_features(e) for e in elems]

    # net -> list of element indices that touch it (skip ground)
    net_members: dict[str, list[int]] = {}
    for e in elems:
        for n in e.nodes:
            if n in GROUND_NETS:
                continue
            net_members.setdefault(n, []).append(idx[e.name])

    src: list[int] = []
    dst: list[int] = []
    seen = set()
    for members in net_members.values():
        for a in members:
            for b in members:
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                src.append(a)
                dst.append(b)
    # degree feature appended per node (number of distinct neighbors)
    deg = [0] * len(elems)
    for a, b in zip(src, dst):
        deg[a] += 1
    for i in range(len(x)):
        x[i] = x[i] + [deg[i] / 8.0]   # normalized degree

    return {"x": x, "edge_index": [src, dst], "node_names": names}


def graph_from_cir(cir_text: str) -> dict:
    return graph_from_netlist(parse_cir(cir_text))


# full node-feature width including the appended degree channel
FEAT_DIM = NODE_DIM + 1


def graph_summary(g: dict) -> list[float]:
    """A fixed-length graph-level feature vector (for a quick MLP baseline): node
    count, edge count, and per-type counts + mean numeric channels."""
    x = g["x"]
    n = len(x)
    n_edges = len(g["edge_index"][0]) / 2.0   # undirected pairs counted twice
    type_counts = [0.0] * len(TYPES)
    sums = [0.0] * (FEAT_DIM - len(TYPES))
    for row in x:
        for i in range(len(TYPES)):
            type_counts[i] += row[i]
        for j in range(len(TYPES), FEAT_DIM):
            sums[j - len(TYPES)] += row[j]
    means = [s / n if n else 0.0 for s in sums]
    return [n / 20.0, n_edges / 20.0] + type_counts + means
