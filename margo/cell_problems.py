"""
cell_problems.py — express each clean-room cell as a cell-agnostic CellProblem so
the generic inverse-design engine (gensearch.py) can verify + optimise it.

Each factory returns a ``CellProblem`` bundling the builder, the defining truth
table (named stimuli + expected output pulse counts), and the tunable knobs. This
is the single place a new cell type (TFF, DCSFQ, ...) is registered.

Clean-IP: our own builders + our own truth tables; no third-party cell libraries used.
"""
from __future__ import annotations

from .builders import (
    JTLSpec,
    SplitterSpec,
    build_jtl_testbench,
    build_splitter_testbench,
)
from .comparator import CLK1_T, CLK2_T, D_T, CompDFFSpec, MKNOBS, build_comp_dff
from .and2 import A_T, B_T, CLK_T, build_and2
from .gensearch import CellProblem, Experiment


# --------------------------------------------------------------------------- #
# Clocked-comparator D flip-flop (the cell, now as a CellProblem)
# --------------------------------------------------------------------------- #


def comp_dff_problem(base: CompDFFSpec | None = None) -> CellProblem:
    """The clocked-comparator DFF as a generic problem. The defining DFF truth table:
    clk-only -> 0 (read 0), data+clk -> 1 (read 1), data+clk+clk -> 1 (destructive
    re-read). Output is junction BF. Knobs are the comparator's MKNOBS."""
    base = base or CompDFFSpec()

    def build(spec: CompDFFSpec, stim: dict) -> NetlistGraph:  # type: ignore[name-defined]
        return build_comp_dff(spec, stim["data"], stim["clk"])

    experiments = [
        Experiment("read0", {"data": [], "clk": [CLK1_T]}, {"BF": 0}),
        Experiment("read1", {"data": [D_T], "clk": [CLK1_T]}, {"BF": 1}),
        Experiment("reread", {"data": [D_T], "clk": [CLK1_T, CLK2_T]}, {"BF": 1}),
    ]
    return CellProblem(base_spec=base, knobs=MKNOBS, build_fn=build,
                       experiments=experiments, bias_prefixes=("IB",))


# --------------------------------------------------------------------------- #
# Clocked AND2 / OR2 (the keystone combinational gates — DFF-siblings)
# --------------------------------------------------------------------------- #

# The coincidence cell is the SAME two-loop comparator; the forward-junction bias
# bias_f_ua IS the threshold: low (~40uA) => both loops needed (AND), higher
# (~70uA) => one loop suffices (OR). bias_q sets shared-node leakage.
GATE_KNOBS = {
    "bias_f_ua":  (30.0, 90.0, 5.0),
    "bias_q_ua":  (80.0, 150.0, 10.0),
    "bias_in_ua": (110.0, 170.0, 10.0),
    "lq_ph":      (8.0, 24.0, 2.0),
}

# Canonical operating points found by the and2.py sweep (centres of the functional
# regions verified in JoSIM: AND2 bf in [30,50], OR2 bf in [60,80]).
AND2_SPEC = CompDFFSpec(bias_f_ua=40.0, bias_q_ua=90.0)
OR2_SPEC = CompDFFSpec(bias_f_ua=70.0, bias_q_ua=120.0)


def _gate_build(spec: CompDFFSpec, stim: dict):  # type: ignore[name-defined]
    return build_and2(spec, stim["a"], stim["b"], stim["clk"])


def and2_problem(base: CompDFFSpec | None = None) -> CellProblem:
    """Clocked AND2 (coincidence gate). Truth table on output BF: only when BOTH
    data inputs are stored does the clock read out a 1; one or none -> 0.
        00->0   10->0   01->0   11->1
    The forward-junction bias is tuned LOW so a single stored loop cannot tip BF."""
    base = base or AND2_SPEC
    experiments = [
        Experiment("and00", {"a": [], "b": [], "clk": [CLK_T]}, {"BF": 0}),
        Experiment("and10", {"a": [A_T], "b": [], "clk": [CLK_T]}, {"BF": 0}),
        Experiment("and01", {"a": [], "b": [B_T], "clk": [CLK_T]}, {"BF": 0}),
        Experiment("and11", {"a": [A_T], "b": [B_T], "clk": [CLK_T]}, {"BF": 1}),
    ]
    return CellProblem(base_spec=base, knobs=GATE_KNOBS, build_fn=_gate_build,
                       experiments=experiments, bias_prefixes=("IB",))


def or2_problem(base: CompDFFSpec | None = None) -> CellProblem:
    """Clocked OR2 — the SAME two-loop coincidence cell with the threshold tuned so
    a single stored loop is enough. Truth table on output BF:
        00->0   10->1   01->1   11->1
    Demonstrates AND and OR are one tunable cell (bias_f is the coincidence knob)."""
    base = base or OR2_SPEC
    experiments = [
        Experiment("or00", {"a": [], "b": [], "clk": [CLK_T]}, {"BF": 0}),
        Experiment("or10", {"a": [A_T], "b": [], "clk": [CLK_T]}, {"BF": 1}),
        Experiment("or01", {"a": [], "b": [B_T], "clk": [CLK_T]}, {"BF": 1}),
        Experiment("or11", {"a": [A_T], "b": [B_T], "clk": [CLK_T]}, {"BF": 1}),
    ]
    return CellProblem(base_spec=base, knobs=GATE_KNOBS, build_fn=_gate_build,
                       experiments=experiments, bias_prefixes=("IB",))


# --------------------------------------------------------------------------- #
# 1->2 SFQ splitter (fanout cell — the structurally-different 2nd cell type)
# --------------------------------------------------------------------------- #

# Knobs the search may perturb. bias_br_ua is the fanout-critical knob: too low and
# a branch can't reach Ic, so it drops its output pulse and the fanout breaks.
SPLITTER_KNOBS = {
    "bias_br_ua":   (120.0, 180.0, 5.0),
    "bias_in_ua":   (120.0, 180.0, 5.0),
    "l_branch_ph":  (1.5, 3.5, 0.25),
}


def splitter_problem(base: SplitterSpec | None = None) -> CellProblem:
    """The 1->2 SFQ splitter as a generic problem — a fanout cell with a REAL margin
    (branch bias must keep both branches above Ic). Defining truth table: a quiet
    input fires nothing; ONE input fluxon produces ONE pulse on each branch; TWO
    inputs produce two each. Output junctions are the branch junctions BA and BB
    (BIN is the input/receiver). Stimulus is the list of input-pulse times.

    This is the structurally-different 2nd cell that demonstrates the
    cell-agnostic engine generalizes beyond the comparator DFF."""
    base = base or SplitterSpec()

    def build(spec: SplitterSpec, stim: list) -> NetlistGraph:  # type: ignore[name-defined]
        return build_splitter_testbench(spec, pulse_ps=stim)

    experiments = [
        Experiment("quiet", [], {"BIN": 0, "BA": 0, "BB": 0}),
        Experiment("one", [200.0], {"BIN": 1, "BA": 1, "BB": 1}),
        Experiment("two", [200.0, 400.0], {"BIN": 2, "BA": 2, "BB": 2}),
    ]
    return CellProblem(base_spec=base, knobs=SPLITTER_KNOBS, build_fn=build,
                       experiments=experiments, bias_prefixes=("IB",))


# --------------------------------------------------------------------------- #
# Josephson transmission line (ballistic propagation — the robust baseline cell)
# --------------------------------------------------------------------------- #

# Continuous knobs the search/sweep may perturb. n_jj is a STRUCTURAL parameter
# (it changes the junction count/names) so it is varied at the corpus level, not here.
JTL_KNOBS = {
    "bias_ua":      (90.0, 180.0, 5.0),
    "l_series_ph":  (1.5, 4.0, 0.25),
    "area":         (1.6, 2.8, 0.1),
}


def jtl_problem(base: JTLSpec | None = None) -> CellProblem:
    """A Josephson transmission line as a generic problem. Defining truth table: a
    quiet input fires nothing; each injected SFQ pulse propagates through EVERY
    junction, so the input junction B01 and the output junction B{n_jj} both slip
    once per input pulse. Stimulus is the list of input-pulse times.

    The expected counts reference ``base.n_jj`` (the output junction), so a JTL of
    any length registers correctly. bias/series-L/area are the tunable knobs."""
    base = base or JTLSpec()
    out_jj = f"B{base.n_jj:02d}"

    def build(spec: JTLSpec, stim: list) -> NetlistGraph:  # type: ignore[name-defined]
        return build_jtl_testbench(spec, pulse_ps=stim)

    experiments = [
        Experiment("quiet", [], {"B01": 0, out_jj: 0}),
        Experiment("one", [200.0], {"B01": 1, out_jj: 1}),
        Experiment("three", [100.0, 300.0, 500.0], {"B01": 3, out_jj: 3}),
    ]
    return CellProblem(base_spec=base, knobs=JTL_KNOBS, build_fn=build,
                       experiments=experiments, bias_prefixes=("IB",))
