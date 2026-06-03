"""
search_accel.py — surrogate-accelerated inverse-design search.

The pure engine (``gensearch.search_margin``) is honest but slow: every candidate it
considers costs a full JoSIM margin-walk (many simulations). That is fine for a demo
on one fast cell but does not scale to a real design loop.

This module composes the trained surrogate with the JoSIM engine to get the
best of both — "surrogate proposes, JoSIM disposes":

    1. PROPOSE (cheap)  : draw a large pool of candidate specs, score each by the
                          surrogate's predicted bias-margin width in ~microseconds,
                          and keep the top ``finalists`` — ZERO JoSIM calls.
    2. DISPOSE (honest) : run the REAL JoSIM ``margin_fitness`` on ONLY those
                          finalists and return the best JoSIM-verified design.

The surrogate never decides functionality or the reported margin — JoSIM does. The
surrogate only decides which handful of candidates are worth the JoSIM time. We count
JoSIM simulations both ways to quantify the speed unlock.

Clean-IP: our own engine + our own trained head + our own builders.
"""
from __future__ import annotations

import contextlib
import random
import time
from dataclasses import replace

from . import gensearch as gs
from .surrogate import NetlistSurrogate


@contextlib.contextmanager
def count_josim():
    """Count JoSIM simulations executed by the gensearch engine while active, by
    wrapping the ``simulate`` symbol gensearch resolves at call time."""
    orig = gs.simulate
    counter = {"n": 0}

    def wrapped(*a, **k):
        counter["n"] += 1
        return orig(*a, **k)

    gs.simulate = wrapped
    try:
        yield counter
    finally:
        gs.simulate = orig


def _random_spec(problem, rng):
    state = {k: rng.choice(gs._knob_grid(lo, hi, st))
             for k, (lo, hi, st) in problem.knobs.items()}
    return replace(problem.base_spec, **state)


def surrogate_search(problem, surrogate: NetlistSurrogate, pool: int = 300,
                     finalists: int = 8, rng_seed: int = 0, mstep: float = 0.025,
                     log=print):
    """Screen a ``pool`` of random candidates with the surrogate (no JoSIM), then
    JoSIM-verify the top ``finalists`` with the real margin fitness. Returns
    (best_spec, best_fitness, best_detail, n_josim_sims, predicted_width)."""
    rng = random.Random(rng_seed)
    candidates = [problem.base_spec] + [_random_spec(problem, rng) for _ in range(pool - 1)]

    t0 = time.perf_counter()
    scored = [(surrogate.predict_spec(problem, c), c) for c in candidates]
    screen_s = time.perf_counter() - t0
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:finalists]
    log(f"  surrogate screened {pool} candidates in {screen_s*1e3:.1f} ms "
        f"({screen_s/pool*1e6:.1f} us/candidate); top predicted width="
        f"{top[0][0]:.3f}")

    best = None  # (fitness, spec, detail, predicted)
    with count_josim() as counter:
        for pred, c in top:
            fit, det = gs.margin_fitness(problem, c, mstep)
            if best is None or fit > best[0]:
                best = (fit, c, det, pred)
            log(f"    finalist pred={pred:.3f} -> JoSIM fitness={fit:.3f} {det}")
    return best[1], best[0], best[2], counter["n"], best[3]


def compare(problem, surrogate, pool=300, finalists=8, restarts=3, max_evals=120,
            rng_seed=0, mstep=0.05, log=print):
    """Run the pure JoSIM hill-climb and the surrogate-accelerated search on the same
    problem and report best margin + JoSIM-sim count for each (the speed unlock)."""
    log("== PURE JoSIM search (gensearch.search_margin) ==")
    with count_josim() as c_pure:
        t0 = time.perf_counter()
        _, pure_fit, pure_det = gs.search_margin(
            problem, restarts=restarts, max_evals=max_evals, rng_seed=rng_seed,
            mstep=mstep, log=lambda *_: None)
        pure_s = time.perf_counter() - t0
    log(f"  pure: best margin={pure_fit:.3f} {pure_det}  "
        f"JoSIM_sims={c_pure['n']}  wall={pure_s:.1f}s")

    log("== SURROGATE-ACCELERATED search (surrogate proposes, JoSIM disposes) ==")
    t0 = time.perf_counter()
    _, acc_fit, acc_det, acc_sims, pred = surrogate_search(
        problem, surrogate, pool=pool, finalists=finalists, rng_seed=rng_seed,
        mstep=mstep, log=log)
    acc_s = time.perf_counter() - t0
    log(f"  accel: best margin={acc_fit:.3f} {acc_det}  "
        f"JoSIM_sims={acc_sims}  wall={acc_s:.1f}s  (surrogate predicted {pred:.3f})")

    if acc_sims > 0:
        log(f"== JoSIM-sim reduction: {c_pure['n']} -> {acc_sims} "
            f"({c_pure['n']/acc_sims:.1f}x fewer)  "
            f"margin pure={pure_fit:.3f} vs accel={acc_fit:.3f} ==")
    return pure_fit, acc_fit, c_pure["n"], acc_sims


def main():
    import argparse
    from .cell_problems import jtl_problem, splitter_problem, comp_dff_problem

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cell", default="jtl", choices=["jtl", "splitter", "dff"])
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--finalists", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=120)
    ap.add_argument("--mstep", type=float, default=0.05)
    args = ap.parse_args()

    problems = {"jtl": jtl_problem, "splitter": splitter_problem, "dff": comp_dff_problem}
    problem = problems[args.cell]()
    surrogate = NetlistSurrogate(args.ckpt)
    print(f"loaded surrogate (target={surrogate.target}, val_MAE={surrogate.val_mae}) "
          f"cell={args.cell}")
    compare(problem, surrogate, pool=args.pool, finalists=args.finalists,
            restarts=args.restarts, max_evals=args.max_evals, mstep=args.mstep)


if __name__ == "__main__":
    main()
