"""
and2.py — a clocked AND2 gate: the keystone combinational cell (DFF-sibling).

WHY THIS CELL: {AND, OR, DFF} is a functionally-complete logic set. The DFF already
works (comparator.py); an RSFQ AND2 is its sibling — a clocked COINCIDENCE comparator
with TWO storage loops feeding one decision node, tuned so the output (forward)
junction fires ONLY when the clock finds BOTH loops charged. If this falls quickly,
"build something that computes" moves from blocked to merely unbuilt.

TOPOLOGY (two storage loops -> shared decision node D -> clocked comparator):

    dataA --feeder--> inA --[BINA->0]   set-A         loop A: inA-LQA-D-BQ-gnd-BINA
                      inA --LQA--> D
    dataB --feeder--> inB --[BINB->0]   set-B         loop B: inB-LQB-D-BQ-gnd-BINB
                      inB --LQB--> D
                      D   --[BQ->0]      shared quantizer at the decision node
    clk   --feeder--> ck  --[BESC->0]   clock escape (read-0 sink)
                      ck  --LCK--> cc --[BC]--> D      clock decision junction
                      D   --LF--> o --[BF->0]          FORWARD/output junction
                      o   --LOUT--> ot --[BOUT->0] --> OUT

The two loops both inject circulating current into node D. The forward junction BF
is biased LOW enough that the clock + ONE stored loop cannot tip it (the clock
escapes via BESC -> output 0), but the clock + BOTH loops' combined current CAN
(-> output 1). The threshold living between "one loop" and "two loops" is the AND.

DEFINING TRUTH TABLE (output = BF phase slips), the honest 4-row pass criterion:
    00 + clk -> 0      10 + clk -> 0      01 + clk -> 0      11 + clk -> 1
Anything that always-fires or always-stays-silent is not an AND.

Clean-IP: reuses our own comparator helpers (_shunt_jj/_fed_input) + JoSIM as judge.
Clean-room; no third-party cell libraries used.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from margo.netlist import drc, emit_cir
from margo.netlist.graph import NetlistGraph, RawLine
from margo.netlist.verifier import simulate

from .builders import _ind, _res, jmitll_model
from .comparator import CompDFFSpec, _fed_input, _shunt_jj, D_T, CLK1_T

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

A_T = D_T          # data-A pulse time (200 ps)
B_T = D_T          # data-B pulse time (200 ps) — both present together for "11"
CLK_T = CLK1_T     # clock readout (320 ps)


def build_and2(spec: CompDFFSpec, a_ts: list[float], b_ts: list[float],
               clk_ts: list[float], t_end_ps: Optional[float] = None) -> NetlistGraph:
    """Build the clocked-AND2 deck. ``a_ts``/``b_ts`` are the data-A/data-B feeder
    pulse times (empty = that input absent); ``clk_ts`` the clock times. Output = BF."""
    all_t = (a_ts or []) + (b_ts or []) + (clk_ts or [])
    if t_end_ps is None:
        t_end_ps = (max(all_t) + 250.0) if all_t else 500.0
    g = NetlistGraph()
    g.add(jmitll_model())

    # --- storage loop A: feeder -> inA -> [BINA], inA-LQA-> D ---
    if a_ts:
        _fed_input(g, "A", "inA", a_ts, spec)
    _shunt_jj(g, "INA", "inA", "0", spec.area, spec.bias_in_ua, spec)
    g.add(_ind("LQA", "inA", "D", spec.lq_ph))

    # --- storage loop B: feeder -> inB -> [BINB], inB-LQB-> D ---
    if b_ts:
        _fed_input(g, "B", "inB", b_ts, spec)
    _shunt_jj(g, "INB", "inB", "0", spec.area, spec.bias_in_ua, spec)
    g.add(_ind("LQB", "inB", "D", spec.lq_ph))

    # --- shared quantizer at the decision node D ---
    _shunt_jj(g, "Q", "D", "0", spec.area, spec.bias_q_ua, spec)

    # --- clock chain: feeder -> ck; escape BESC at ck, decision BC one L away ---
    if clk_ts:
        _fed_input(g, "K", "ck", clk_ts, spec, trigger_ua=spec.feed_clk_trigger_ua)
    _shunt_jj(g, "ESC", "ck", "0", spec.area, spec.bias_esc_ua, spec)
    g.add(_ind("LCK", "ck", "cc", spec.lck_ph))
    _shunt_jj(g, "C", "cc", "D", spec.area, spec.bias_c_ua, spec)

    # --- forward / output: D -> LF -> o -> BF -> 0, tapped to OUT ---
    g.add(_ind("LF", "D", "o", spec.lf_ph))
    _shunt_jj(g, "F", "o", "0", spec.area, spec.bias_f_ua, spec)
    g.add(_ind("LOUT", "o", "ot", spec.lout_ph))
    _shunt_jj(g, "OUT", "ot", "0", spec.area, spec.bias_out_ua, spec)
    g.add(_ind("LOO", "ot", "OUT", spec.l_in_ph))
    g.add(_res("ROUT", "OUT", "0", spec.r_out_ohm))

    g.add(RawLine(text=f".tran 0.1p {t_end_ps:g}p 0 0.1p"))
    for jj in ("BINA", "BINB", "BQ", "BESC", "BC", "BF", "BOUT"):
        g.add(RawLine(text=f".print PHASE {jj}"))
    g.add(RawLine(text=".end"))
    return g


# --------------------------------------------------------------------------- #
# Truth-table oracle
# --------------------------------------------------------------------------- #

# The 4 input combinations (A present?, B present?) -> expected output.
CASES = [
    ("00", [], [], 0),
    ("10", [A_T], [], 0),
    ("01", [], [B_T], 0),
    ("11", [A_T], [B_T], 1),
]


def _out(spec: CompDFFSpec, a_ts, b_ts, which="BF", timeout: int = 120) -> Optional[int]:
    deck = build_and2(spec, a_ts, b_ts, [CLK_T])
    if not drc.check(deck).ok:
        return None
    res = simulate(deck, timeout=timeout)
    if not res.ok:
        return None
    return res.pulses(which)


def truth_table(spec: CompDFFSpec, timeout: int = 120):
    """Run the 4 defining AND2 experiments. Returns (score 0..4, detail, counts)."""
    counts = {}
    score = 0
    for name, a, b, want in CASES:
        n = _out(spec, a, b, timeout=timeout)
        counts[name] = n
        if n == want:
            score += 1
    detail = " ".join(f"{name}={counts[name]}(w{want})" for name, _, _, want in CASES)
    return score, detail, counts


def diagnose(spec: CompDFFSpec, timeout: int = 120, log=print) -> None:
    """Per-junction pulse counts for each input combo — to see WHICH junction wins."""
    log(f"  biases in/q/esc/c/f={spec.bias_in_ua}/{spec.bias_q_ua}/{spec.bias_esc_ua}"
        f"/{spec.bias_c_ua}/{spec.bias_f_ua}  lq={spec.lq_ph}pH")
    for name, a, b, want in CASES:
        deck = build_and2(spec, a, b, [CLK_T])
        if not drc.check(deck).ok:
            log(f"    {name} DRC FAIL"); continue
        res = simulate(deck, timeout=timeout)
        if not res.ok:
            log(f"    {name} sim {res.status}"); continue
        cnt = {j: res.pulses(j) for j in
               ("BINA", "BINB", "BQ", "BESC", "BC", "BF", "BOUT")}
        log(f"    {name}(want {want}) " + " ".join(f"{k}={v}" for k, v in cnt.items() if v))


# --------------------------------------------------------------------------- #
# Find a working point: the AND lives in the bias_f / bias_q balance
# --------------------------------------------------------------------------- #


def sweep(spec: CompDFFSpec, timeout: int = 120, log=print):
    """Coarse 2-D sweep over the coincidence-threshold knobs (bias_f sets how much
    stored current BF needs; bias_q sets how easily the shared node leaks). Returns
    the first spec scoring 4/4, or the best partial."""
    best = (None, -1, "")
    for bf in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0):
        for bq in (90.0, 110.0, 130.0, 150.0):
            s = replace(spec, bias_f_ua=bf, bias_q_ua=bq)
            score, detail, _ = truth_table(s, timeout=timeout)
            tag = "  <-- 4/4" if score == 4 else ""
            log(f"  bf={bf:4.0f} bq={bq:4.0f}  score={score}/4  {detail}{tag}")
            if score > best[1]:
                best = (s, score, detail)
            if score == 4:
                return s, 4, detail
    return best


def main():
    print("== clocked AND2 (DFF-sibling coincidence gate) — JoSIM truth table ==")
    os.makedirs(OUT_DIR, exist_ok=True)

    base = CompDFFSpec()
    print("nominal (DFF defaults):")
    score, detail, _ = truth_table(base)
    print(f"  score={score}/4  {detail}")
    if score < 4:
        print("diagnose nominal:")
        diagnose(base)
        print("sweep coincidence knobs (bias_f x bias_q):")
        spec, score, detail = sweep(base)
    else:
        spec = base

    if score == 4:
        print(f"== AND2 FUNCTIONAL 4/4: {detail} ==")
        deck = build_and2(spec, [A_T], [B_T], [CLK_T])
        path = os.path.join(OUT_DIR, "and2.cir")
        with open(path, "w") as f:
            f.write(emit_cir(deck))
        print(f"   saved {path}  ({len(deck.elements)} elements)  "
              f"bias_f={spec.bias_f_ua} bias_q={spec.bias_q_ua} lq={spec.lq_ph}")
    else:
        print(f"== best so far {score}/4 — checkpoint (may be the comparator-balance wall) ==")


if __name__ == "__main__":
    main()
