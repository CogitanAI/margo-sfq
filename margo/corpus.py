"""
corpus.py — build a topology-faithful, JoSIM-labeled cell corpus.

Sweeps the clean-room parametric builders (JTL, splitter, DFF — all real
``NetlistGraph`` cells with real connectivity) and labels every variant with JoSIM
ground truth via the cell-agnostic oracle (gensearch):

    functional?  +  bias-margin band  +  Ic-margin band

Each record stores the spec, the labels, AND the runnable ``.cir`` netlist (the input
representation the surrogate ingests). This corpus is what the netlist-graph converter
and the trainer build on.

Clean-room: our own builders + our own JoSIM oracle. JoSIM runs via WSL on Windows.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from typing import Any

from margo.netlist import emit_cir

from . import gensearch as gs
from .builders import JTLSpec, SplitterSpec
from .comparator import CompDFFSpec, MKNOBS
from .cell_problems import (
    JTL_KNOBS,
    SPLITTER_KNOBS,
    comp_dff_problem,
    jtl_problem,
    splitter_problem,
)

# Per-cell: the factory that wraps a spec as a CellProblem, and the experiment whose
# deck we save as the honest .cir (the canonical 'functional demonstration').
PROBLEM_FNS = {"jtl": jtl_problem, "splitter": splitter_problem, "dff": comp_dff_problem}
CANON_EXP = {"jtl": "one", "splitter": "one", "dff": "read1"}

CORPUS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "netlist_corpus")

# Corpus sampling ranges are DELIBERATELY WIDER than the search knobs. The search
# (gensearch) only explores the functional region, but the corpus must STRADDLE the
# functional edge so the head has real pass/fail examples — otherwise every label is
# "functional" and a classifier learns nothing. We widen each cell's most
# failure-sensitive knob (bias current) well below its known-good floor so a fraction
# of variants fall off the edge into non-functional territory.
JTL_CORPUS_RANGES = dict(JTL_KNOBS, bias_ua=(40.0, 180.0, 5.0))
SPLITTER_CORPUS_RANGES = dict(
    SPLITTER_KNOBS, bias_in_ua=(60.0, 180.0, 5.0), bias_br_ua=(60.0, 180.0, 5.0)
)

# Per-cell margin-walk step. The DFF's functional band is ~10x narrower than
# JTL/splitter, so the default 0.1 step quantizes its margin to just {0.0,0.1,0.2} —
# a near-useless regression target. A finer 0.05 step resolves the DFF band into a
# real continuum; the robust cells stay at the coarser (faster) step.
MSTEP_BY_CELL = {"jtl": 0.1, "splitter": 0.1, "dff": 0.05}


def _grid(lo: float, hi: float, st: float) -> list[float]:
    return [round(lo + i * st, 6) for i in range(int(round((hi - lo) / st)) + 1)]


def _sample_jtl(rng: random.Random) -> JTLSpec:
    """A random JTL variant: structural n_jj plus continuous knobs sampled from the
    WIDE corpus ranges. The widened bias range deliberately includes non-functional
    points (very low bias) so the corpus carries the functional boundary, not just the
    good region."""
    n_jj = rng.choice([2, 3, 4])
    vals = {k: rng.choice(_grid(lo, hi, st))
            for k, (lo, hi, st) in JTL_CORPUS_RANGES.items()}
    return JTLSpec(n_jj=n_jj, **vals)


def _sample_splitter(rng: random.Random) -> SplitterSpec:
    vals = {k: rng.choice(_grid(lo, hi, st))
            for k, (lo, hi, st) in SPLITTER_CORPUS_RANGES.items()}
    return SplitterSpec(**vals)


def _sample_dff(rng: random.Random) -> CompDFFSpec:
    """A DFF variant sampled in a WINDOW around the known-good base (the 25%-margin
    point). The comparator DFF is balance-critical, so fully-random knobs would be
    almost all non-functional; perturbing each knob by a few grid steps yields mostly
    functional cells with varying margins plus some near-boundary failures — exactly
    the regression + classification signal we want."""
    base = CompDFFSpec()
    vals = {}
    for k, (lo, hi, st) in MKNOBS.items():
        b = getattr(base, k)
        # Widened from +/-2 to +/-4 grid steps: the comparator DFF is balance-critical,
        # so a wider perturbation pushes a meaningful fraction of variants past the
        # functional edge (non-functional examples) while still keeping a functional
        # core with varying margins.
        delta = rng.choice([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4]) * st
        vals[k] = min(hi, max(lo, round(b + delta, 6)))
    return replace(base, **vals)


# Margin-walk caps for corpus labeling. WIDER than the search defaults on purpose:
# JTL/splitter are robust cells whose failure edges sit outside [0.5,1.6] — without
# widening, every variant saturates the cap and the labels carry no signal. The
# over-bias self-oscillation edge (and high-Ic edge) are spec-dependent, so widening
# turns a constant label into a real regression target.
BIAS_CAPS = (0.2, 3.0)
IC_CAPS = (0.4, 2.0)


def label_variant(cell_type: str, spec: Any, mstep: float = 0.1) -> dict:
    """Run the JoSIM oracle on one spec and return an honest-label record. The saved
    ``cir`` is the 'one input pulse' experiment deck — a consistent, runnable,
    real-connectivity representation of the cell for the surrogate to ingest."""
    problem = PROBLEM_FNS[cell_type](spec)
    n_pass, n_total, ok, _ = gs.evaluate(problem, spec)
    bl, bh, bw = gs.bias_margin(problem, spec, step=mstep,
                                lo_cap=BIAS_CAPS[0], hi_cap=BIAS_CAPS[1])
    il, ih, iw = gs.ic_margin(problem, spec, step=mstep,
                              lo_cap=IC_CAPS[0], hi_cap=IC_CAPS[1])
    canon = CANON_EXP[cell_type]
    canon_stim = next(e for e in problem.experiments if e.name == canon).stimulus
    cir = emit_cir(problem.build_fn(spec, canon_stim))
    rec = {
        "cell_type": cell_type,
        "spec": asdict(spec),
        "n_jj": getattr(spec, "n_jj", None),
        "n_pass": n_pass,
        "n_total": n_total,
        "functional": bool(ok),
        "bias_margin_lo": bl, "bias_margin_hi": bh, "bias_margin_width": bw,
        "ic_margin_lo": il, "ic_margin_hi": ih, "ic_margin_width": iw,
        "cir": cir,
    }
    return rec


def generate(n_per_cell: int = 15, cells: tuple[str, ...] = ("jtl", "splitter"),
             seed: int = 0, mstep: float = 0.1,
             out_path: str | None = None, log=print,
             n_start: int = 0, append: bool = False) -> str:
    """Generate + JoSIM-label a corpus. Always includes each cell's nominal base spec
    as the first variant, then ``n_per_cell - 1`` random variants. Writes JSONL
    incrementally (so a long run is crash-safe) and returns the output path.

    RESUME SUPPORT: pass ``n_start>0`` + ``append=True`` to continue a partially-built
    part file — variant indices [n_start, n_per_cell) are generated and APPENDED (the
    base spec at i==0 is skipped on resume). A distinct rng stream (seeded by n_start)
    avoids regenerating the same random specs as the original run."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    if out_path is None:
        tag = "_".join(cells)
        out_path = os.path.join(CORPUS_DIR, f"corpus_{tag}_n{n_per_cell}.jsonl")
    rng = random.Random(seed + n_start)
    samplers = {"jtl": _sample_jtl, "splitter": _sample_splitter, "dff": _sample_dff}
    bases = {"jtl": JTLSpec(), "splitter": SplitterSpec(), "dff": CompDFFSpec()}

    n_written = n_func = 0
    with open(out_path, "a" if append else "w") as f:
        for cell_type in cells:
            for i in range(n_start, n_per_cell):
                spec = bases[cell_type] if i == 0 else samplers[cell_type](rng)
                ms = MSTEP_BY_CELL.get(cell_type, mstep)
                try:
                    rec = label_variant(cell_type, spec, mstep=ms)
                except Exception as e:  # noqa: BLE001 — one bad variant must not kill the run
                    log(f"  [{cell_type} {i}] ERROR {type(e).__name__}: {e}")
                    continue
                rec["id"] = f"{cell_type}_{i:04d}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_written += 1
                n_func += int(rec["functional"])
                log(f"  [{rec['id']}] func={rec['functional']} "
                    f"bias_w={rec['bias_margin_width']:.3f} "
                    f"ic_w={rec['ic_margin_width']:.3f} n_jj={rec['n_jj']}")
    log(f"== corpus done: {n_written} variants ({n_func} functional) -> {out_path} ==")
    return out_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    generate(n_per_cell=n)
