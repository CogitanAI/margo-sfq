"""
gensearch.py — cell-agnostic inverse-design engine.

Earlier work proved the closed loop (construct -> DRC -> JoSIM -> widen margin) on ONE
cell, but the machinery lived inside comparator.py hardwired to ``CompDFFSpec`` /
``MKNOBS`` / the comparator's specific 3-experiment truth table and the "BF" output
junction. To scale to more cell types without copying
that machinery per cell, this module lifts the engine into a small, explicit
problem interface:

    CellProblem = (base_spec, knobs, build_fn, experiments, bias_prefixes)

where ``build_fn(spec, stimulus) -> NetlistGraph`` constructs a runnable deck for a
named stimulus, and each ``Experiment`` names the stimulus plus the EXPECTED output
pulse counts on named junctions. A cell is FUNCTIONAL iff every experiment yields
its expected counts in JoSIM ground truth. The bias / Ic margin is the contiguous
scale band (walked outward from nominal) over which ALL experiments still hold.

Design rules carried over from the comparator work:
  * JoSIM is the only judge (never the surrogate) — the caller must NAME the output
    junction and its expected count; the engine never guesses what "correct" means.
  * The margin fitness GATES on functional first (non-functional designs score
    negative, proportional to how many experiments they pass), then rewards the
    bias-margin WIDTH — so the search climbs to functional, then widens.
  * The margin is measured by a fine "walk outward from 1.0" (default 0.025 step),
    which resolves sub-10% bands a coarse grid cannot.

Clean-IP: this is our own engine operating on our own clean-room builders; no
third-party cell-library files are used.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from margo.netlist import drc, emit_cir, parse_cir
from margo.netlist.graph import NetlistGraph
from margo.netlist.verifier import scale_bias_sources, scale_jj_icrit, simulate

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #


@dataclass
class Experiment:
    """One defining experiment for a cell: build the deck for ``stimulus`` and check
    that each named junction slips exactly ``expected[jj]`` times in JoSIM.

    ``stimulus`` is whatever opaque object the cell's ``build_fn`` understands (e.g.
    ``{"data": [...], "clk": [...]}`` for the DFF). ``expected`` maps junction name ->
    required integer pulse count (a 0 means "must NOT fire")."""
    name: str
    stimulus: Any
    expected: dict[str, int]


@dataclass
class CellProblem:
    """A cell-agnostic inverse-design problem.

    base_spec     : the genotype dataclass instance (knob values read via getattr)
    knobs         : {field_name: (lo, hi, step)} the search may perturb
    build_fn      : (spec, stimulus) -> NetlistGraph  (a runnable, probed deck)
    experiments   : the defining truth table; ALL must pass for FUNCTIONAL
    bias_prefixes : source-name prefixes treated as standing bias (for bias margin)
    timeout       : per-sim JoSIM timeout (s)
    """
    base_spec: Any
    knobs: dict[str, tuple[float, float, float]]
    build_fn: Callable[[Any, Any], NetlistGraph]
    experiments: list[Experiment]
    bias_prefixes: tuple[str, ...] = ("IB",)
    timeout: int = 120


# --------------------------------------------------------------------------- #
# Evaluation primitives
# --------------------------------------------------------------------------- #


def _counts_match(res_pulses: Callable[[str], int], expected: dict[str, int]) -> bool:
    return all(res_pulses(jj) == cnt for jj, cnt in expected.items())


def run_experiment(problem: CellProblem, spec: Any, exp: Experiment,
                   mutate: Optional[Callable[[NetlistGraph, float], NetlistGraph]] = None,
                   factor: float = 1.0) -> Optional[bool]:
    """Build + (optionally scale) + DRC + simulate one experiment. Returns True if
    the JoSIM pulse counts match ``exp.expected``, False if they don't, or None on a
    DRC/sim failure (so the search can penalise broken regions distinctly)."""
    deck = problem.build_fn(spec, exp.stimulus)
    if mutate is not None and abs(factor - 1.0) > 1e-9:
        deck = mutate(deck, factor)
    if not drc.check(deck).ok:
        return None
    res = simulate(deck, timeout=problem.timeout)
    if not res.ok:
        return None
    return _counts_match(res.pulses, exp.expected)


def evaluate(problem: CellProblem, spec: Any,
             mutate: Optional[Callable] = None, factor: float = 1.0):
    """Score a spec: how many experiments pass (out of all). Returns
    (n_pass, n_total, all_ok, detail). ``all_ok`` is True iff every experiment
    passed (None counts as a fail)."""
    results = []
    n_pass = 0
    for exp in problem.experiments:
        ok = run_experiment(problem, spec, exp, mutate, factor)
        results.append((exp.name, ok))
        if ok is True:
            n_pass += 1
    n_total = len(problem.experiments)
    return n_pass, n_total, n_pass == n_total, results


def is_functional(problem: CellProblem, spec: Any) -> bool:
    return evaluate(problem, spec)[2]


# --------------------------------------------------------------------------- #
# Margin measurement (walk outward from nominal)
# --------------------------------------------------------------------------- #


def _margin_walk(problem: CellProblem, spec: Any,
                 mutate: Callable[[NetlistGraph, float], NetlistGraph],
                 step: float, lo_cap: float, hi_cap: float):
    """Confirm the FULL truth table holds at nominal (factor 1.0), then walk OUTWARD
    on each side, stopping at the first factor where ANY experiment fails. Returns
    (low, high, width); (None, None, 0.0) if not functional at nominal."""
    if not evaluate(problem, spec, mutate, 1.0)[2]:
        return None, None, 0.0
    low = 1.0
    f = round(1.0 - step, 4)
    while f >= lo_cap - 1e-9 and evaluate(problem, spec, mutate, f)[2]:
        low = f
        f = round(f - step, 4)
    high = 1.0
    f = round(1.0 + step, 4)
    while f <= hi_cap + 1e-9 and evaluate(problem, spec, mutate, f)[2]:
        high = f
        f = round(f + step, 4)
    return low, high, high - low


def bias_margin(problem: CellProblem, spec: Any, step: float = 0.025,
                lo_cap: float = 0.5, hi_cap: float = 1.6):
    """Global bias margin: scale every standing-bias source (names matching
    ``problem.bias_prefixes``) and find the band where the whole truth table holds.
    Signal/trigger sources are untouched, so this is the textbook RSFQ bias margin."""
    mutate = lambda g, f: scale_bias_sources(g, f, problem.bias_prefixes)  # noqa: E731
    return _margin_walk(problem, spec, mutate, step, lo_cap, hi_cap)


def ic_margin(problem: CellProblem, spec: Any, step: float = 0.025,
              lo_cap: float = 0.6, hi_cap: float = 1.4):
    """Global Ic margin: scale every junction's critical current (process/fab
    robustness) and find the band where the whole truth table holds."""
    return _margin_walk(problem, spec, scale_jj_icrit, step, lo_cap, hi_cap)


# --------------------------------------------------------------------------- #
# Fitness + search
# --------------------------------------------------------------------------- #


def margin_fitness(problem: CellProblem, spec: Any, mstep: float = 0.025):
    """Inverse-design fitness: hard-gate on the full truth table, then reward the
    bias-margin WIDTH. Non-functional designs score negative (``-10 + n_pass``) so the
    search first climbs to functional, then widens the margin. Higher is better."""
    n_pass, n_total, ok, _ = evaluate(problem, spec)
    if not ok:
        return -10.0 + n_pass, ("nonfunc", n_pass, n_total)
    lo, hi, w = bias_margin(problem, spec, step=mstep)
    w = w if w is not None else 0.0
    return w, ("func", lo, hi, w)


def _knob_grid(lo: float, hi: float, st: float) -> list[float]:
    return [lo + i * st for i in range(int(round((hi - lo) / st)) + 1)]


def search_margin(problem: CellProblem, restarts: int = 5, max_evals: int = 600,
                  rng_seed: int = 0, mstep: float = 0.025, log=print):
    """Coordinate-ascent hill-climb (with random restarts) MAXIMISING margin_fitness
    over ``problem.knobs``. Restart 0 starts from ``problem.base_spec`` (inside the
    basin if it is already functional and only widens the margin); later restarts
    start from a random knob assignment. Returns (best_spec, best_fitness, detail)."""
    rng = random.Random(rng_seed)
    base = problem.base_spec
    evals = 0

    def fit(state: dict):
        nonlocal evals
        evals += 1
        return margin_fitness(problem, replace(base, **state), mstep)

    def random_state():
        return {k: rng.choice(_knob_grid(lo, hi, st))
                for k, (lo, hi, st) in problem.knobs.items()}

    best = None  # (fitness, state, detail)
    seed_state = {k: getattr(base, k) for k in problem.knobs}
    for r in range(restarts):
        state = seed_state if r == 0 else random_state()
        cur_fit, cur_det = fit(state)
        improved = True
        while improved and evals < max_evals:
            improved = False
            for k in problem.knobs:
                lo, hi, st = problem.knobs[k]
                for delta in (st, -st):
                    if evals >= max_evals:
                        break
                    nv = max(lo, min(hi, state[k] + delta))
                    if nv == state[k]:
                        continue
                    trial = dict(state)
                    trial[k] = nv
                    f, det = fit(trial)
                    if f > cur_fit:
                        state, cur_fit, cur_det = trial, f, det
                        improved = True
                        break
            if best is None or cur_fit > best[0]:
                best = (cur_fit, dict(state), cur_det)
                log(f"  [r{r} ev{evals}] fit={cur_fit:.3f} {cur_det} "
                    + " ".join(f"{k}={state[k]:g}" for k in problem.knobs))
        log(f"  -- restart {r} done (evals={evals}, best={best[0]:.3f}) --")
        if evals >= max_evals:
            break
    best_fit, best_state, best_det = best
    best_spec = replace(base, **best_state)
    log(f"== margin search done: best fitness={best_fit:.3f} {best_det} "
        f"evals={evals} ==")
    return best_spec, best_fit, best_det


# --------------------------------------------------------------------------- #
# Artifact save / reload round-trip (cell-agnostic)
# --------------------------------------------------------------------------- #


def save_artifact(problem: CellProblem, spec: Any, exp_name: str, name: str,
                  log=print) -> str:
    """Emit the deck for the named experiment to out/<name>.cir, then
    RE-PARSE it from disk and RE-SIMULATE to prove the artifact round-trips and still
    yields that experiment's expected pulse counts in JoSIM. Returns the path. Works
    for ANY CellProblem (the engine never hard-codes a cell's output junction)."""
    exp = next(e for e in problem.experiments if e.name == exp_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    deck = problem.build_fn(spec, exp.stimulus)
    path = os.path.join(OUT_DIR, f"{name}.cir")
    with open(path, "w") as f:
        f.write(emit_cir(deck))
    log(f"  saved {path}")
    # round-trip: parse back from disk and re-simulate
    reparsed = parse_cir(open(path).read())
    res = simulate(reparsed, timeout=problem.timeout)
    got = {jj: (res.pulses(jj) if res.ok else None) for jj in exp.expected}
    ok = res.ok and _counts_match(res.pulses, exp.expected)
    log(f"  reload+resim [{exp_name}]: got={got} want={exp.expected}  -> "
        f"{'ROUND-TRIP OK' if ok else 'MISMATCH'}")
    return path
