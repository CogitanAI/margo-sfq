"""
verifier.py — JoSIM oracle: NetlistGraph -> JoSIM -> metrics + margins.

This is the ground-truth oracle. A NetlistGraph that is a *complete*
runnable deck (has stimulus + ``.tran``) is emitted, simulated with JoSIM, and
reduced to RSFQ-meaningful metrics (SFQ pulse counts per junction, propagation
delay, peak current). On top of the single-shot oracle sit margin sweeps:

  * ``ic_margin_sweep``   — scale every JJ model's ``icrit`` uniformly, find the
                            contiguous band around nominal where the cell stays
                            functional. This is the Ic (critical-current) margin.
  * ``bias_margin_sweep`` — scale every bias current source's amplitude, find the
                            functional band. This is the bias margin.

JoSIM is a Linux binary; on Windows it runs through WSL. The CSV->metrics reduction
uses ``josim_parse.parse_josim_csv`` (winding-number pulse counting, delay, peak
current) so pulse counting is consistent.

Honesty note: functional checks are convention-light. The caller names the *output
junction* whose switching defines the output; we do not guess. ``max_frequency`` is
intentionally left to a per-cell harness rather than faked here.
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from margo.netlist import emit_cir, units  # noqa: E402
from margo.netlist.graph import JJ, NetlistGraph, Source  # noqa: E402


# --------------------------------------------------------------------------- #
# WSL plumbing
# --------------------------------------------------------------------------- #


def _win_to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def wsl_josim_available() -> bool:
    try:
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "command -v josim-cli"],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class SimResult:
    status: str                       # "ok" | "no_tran" | "fail" | "timeout"
    metrics: dict = field(default_factory=dict)
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def pulses(self, jj_name: str) -> int:
        """SFQ pulse count (winding number) at a junction's phase trace."""
        pc = self.metrics.get("pulse_counts", {})
        return int(pc.get(f"P({jj_name})", pc.get(jj_name, 0)))


# --------------------------------------------------------------------------- #
# Single-shot oracle
# --------------------------------------------------------------------------- #


def _has_tran(g: NetlistGraph) -> bool:
    from margo.netlist.graph import RawLine
    for it in g.items:
        if isinstance(it, RawLine) and it.text.strip().lower().startswith(".tran"):
            return True
    return False


def simulate(graph: NetlistGraph, timeout: int = 300,
             work_dir: Optional[str] = None) -> SimResult:
    """Emit `graph`, run JoSIM via WSL, return parsed metrics.

    `graph` must be a complete runnable deck (stimulus + `.tran`). A bare
    `.subckt` cell returns status 'no_tran' — wrap it in a testbench first.
    """
    if not _has_tran(graph):
        return SimResult(status="no_tran",
                         stderr="graph has no .tran; needs a testbench harness")
    # Lazy import: parse_josim_csv pulls numpy.
    from margo.netlist.josim_parse import parse_josim_csv

    cleanup = work_dir is None
    wd = work_dir or tempfile.mkdtemp(prefix="josim_verify_")
    cir = os.path.join(wd, "deck.cir")
    out_csv = os.path.join(wd, "out.csv")
    try:
        with open(cir, "w") as f:
            f.write(emit_cir(graph))
        cmd = (f"cd {_win_to_wsl(wd)!r} && "
               f"josim-cli -o {_win_to_wsl(out_csv)!r} -V 1 {_win_to_wsl(cir)!r}")
        proc = subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                              capture_output=True, text=True, timeout=timeout)
        if not os.path.exists(out_csv):
            return SimResult(status="fail", stderr=(proc.stderr or "")[:500])
        from pathlib import Path
        metrics = parse_josim_csv(Path(out_csv), testbench_path=Path(cir))
        metrics["status"] = "ok"
        return SimResult(status="ok", metrics=metrics)
    except subprocess.TimeoutExpired:
        return SimResult(status="timeout")
    finally:
        if cleanup:
            for p in (cir, out_csv):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(wd)
            except OSError:
                pass


def is_functional(result: SimResult, output_jj: str,
                  expected_pulses: Optional[int] = None,
                  min_pulses: int = 1) -> bool:
    """Did the named output junction switch as expected?

    With `expected_pulses`, requires an exact SFQ pulse count (the strict gate
    for a known stimulus). Otherwise requires at least `min_pulses` switching
    events (the cell propagated *something*)."""
    if not result.ok:
        return False
    n = result.pulses(output_jj)
    if expected_pulses is not None:
        return n == expected_pulses
    return n >= min_pulses


# --------------------------------------------------------------------------- #
# Knob mutators (operate on a *copy* of the graph)
# --------------------------------------------------------------------------- #


def scale_jj_icrit(graph: NetlistGraph, factor: float) -> NetlistGraph:
    """Return a copy with every jj-model's `icrit` scaled by `factor`.

    Uniformly scaling critical current is the Ic-margin knob; editing the model
    rather than per-JJ `area` works even when areas are symbolic params."""
    g = copy.deepcopy(graph)
    _scale_models_recursive(g, factor)
    return g


def _scale_models_recursive(g: NetlistGraph, factor: float) -> None:
    for m in g.models.values():
        if m.mtype == "jj" and "icrit" in m.params:
            base = units.parse_value(m.params["icrit"])
            m.set_param("icrit", units.format_value(base * factor))
    for sub in g.subckts.values():
        _scale_models_recursive(sub.graph, factor)


_PWL_NAME_RE = re.compile(r"^\s*(pwl|pulse)\s*\((.*)\)\s*$", re.IGNORECASE)


def _scale_pwl_amplitudes(func: str, factor: float) -> Optional[str]:
    """Scale the value (amplitude) tokens of a pwl(...) drive, leaving the time
    tokens untouched. Returns None if `func` isn't a pwl we can rewrite."""
    m = _PWL_NAME_RE.match(func)
    if not m or m.group(1).lower() != "pwl":
        return None
    toks = m.group(2).replace(",", " ").split()
    if len(toks) % 2 != 0:
        return None
    out = []
    for i, t in enumerate(toks):
        if i % 2 == 1:  # value position
            try:
                out.append(units.format_value(units.parse_value(t) * factor))
            except ValueError:
                out.append(t)
        else:
            out.append(t)
    return f"pwl({' '.join(out)})"


def scale_bias_sources(graph: NetlistGraph, factor: float,
                       name_prefixes: tuple[str, ...] = ("IB", "I")) -> NetlistGraph:
    """Return a copy with bias current-source amplitudes scaled by `factor`.

    A bias source is a current source (letter I) whose drive is a pwl ramp; we
    scale its amplitude. `name_prefixes` restricts which sources count as bias
    (default: any I-source; pass ("IB",) to scale only IB-named biases)."""
    g = copy.deepcopy(graph)
    _scale_bias_recursive(g, factor, name_prefixes)
    return g


def _scale_bias_recursive(g: NetlistGraph, factor: float,
                          name_prefixes: tuple[str, ...]) -> None:
    pref = tuple(p.upper() for p in name_prefixes)
    for el in g.elements:
        if isinstance(el, Source) and el.letter == "I" and el.func:
            if not el.name.upper().startswith(pref):
                continue
            scaled = _scale_pwl_amplitudes(el.func, factor)
            if scaled is not None:
                el.func = scaled
    for sub in g.subckts.values():
        _scale_bias_recursive(sub.graph, factor, name_prefixes)


# --------------------------------------------------------------------------- #
# Margin sweeps
# --------------------------------------------------------------------------- #


@dataclass
class MarginResult:
    nominal_ok: bool
    low: Optional[float]      # lowest functional factor
    high: Optional[float]     # highest functional factor
    width: Optional[float]    # high - low (relative margin band)
    samples: list = field(default_factory=list)  # (factor, functional) pairs


def _sweep(graph: NetlistGraph,
           mutate: Callable[[NetlistGraph, float], NetlistGraph],
           ok_fn: Callable[[SimResult], bool],
           lo: float, hi: float, step: float,
           timeout: int) -> MarginResult:
    factors = []
    f = lo
    while f <= hi + 1e-9:
        factors.append(round(f, 4))
        f += step
    if 1.0 not in factors:
        factors.append(1.0)
        factors.sort()

    results: dict[float, bool] = {}
    for fac in factors:
        g = graph if abs(fac - 1.0) < 1e-9 else mutate(graph, fac)
        res = simulate(g, timeout=timeout)
        results[fac] = ok_fn(res)

    nominal_ok = results.get(1.0, False)
    # Contiguous functional band that contains nominal (1.0).
    low = high = None
    if nominal_ok:
        low = high = 1.0
        for fac in sorted((f for f in factors if f < 1.0), reverse=True):
            if results[fac]:
                low = fac
            else:
                break
        for fac in sorted(f for f in factors if f > 1.0):
            if results[fac]:
                high = fac
            else:
                break
    width = (high - low) if (low is not None and high is not None) else None
    return MarginResult(nominal_ok=nominal_ok, low=low, high=high, width=width,
                        samples=sorted(results.items()))


def ic_margin_sweep(graph: NetlistGraph, output_jj: str,
                    expected_pulses: Optional[int] = None,
                    lo: float = 0.7, hi: float = 1.3, step: float = 0.05,
                    timeout: int = 300) -> MarginResult:
    """Ic margin: functional band as JJ critical current scales lo..hi."""
    ok = lambda r: is_functional(r, output_jj, expected_pulses)  # noqa: E731
    return _sweep(graph, scale_jj_icrit, ok, lo, hi, step, timeout)


def bias_margin_sweep(graph: NetlistGraph, output_jj: str,
                      expected_pulses: Optional[int] = None,
                      name_prefixes: tuple[str, ...] = ("IB", "I"),
                      lo: float = 0.7, hi: float = 1.3, step: float = 0.05,
                      timeout: int = 300) -> MarginResult:
    """Bias margin: functional band as bias-source amplitude scales lo..hi."""
    ok = lambda r: is_functional(r, output_jj, expected_pulses)  # noqa: E731
    mutate = lambda g, f: scale_bias_sources(g, f, name_prefixes)  # noqa: E731
    return _sweep(graph, mutate, ok, lo, hi, step, timeout)
