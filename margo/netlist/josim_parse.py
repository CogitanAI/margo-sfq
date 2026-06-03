"""josim_parse.py — parse a JoSIM ``-V 1`` output CSV into SFQ-meaningful metrics.

Counts SFQ pulses per junction as the winding number (2-pi phase slips), extracts
input pulse times from the testbench PWL current sources, and derives a propagation
delay and a functional flag. Pure CSV/PWL parsing; depends only on numpy.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def _extract_input_pulse_times(tb_text: str) -> list[float]:
    """Parse ``I_<name> n+ n- pwl(...)`` current sources; each 0 -> nonzero rising
    edge is one input SFQ pulse. Returns pulse times in ps, sorted."""
    times = []
    for line in tb_text.splitlines():
        m = re.match(r"^\s*I_[a-zA-Z0-9_]+\s+\S+\s+\S+\s+pwl\((.*)\)", line, re.IGNORECASE)
        if not m:
            continue
        toks = m.group(1).strip().replace(",", " ").split()
        pairs = list(zip(toks[0::2], toks[1::2]))
        for i in range(1, len(pairs)):
            t_str, v_str = pairs[i]
            _, v_prev = pairs[i - 1]
            try:
                v_curr = float(v_str.rstrip("uUnNpPfFmMkK"))
                v_prev_f = float(v_prev.rstrip("uUnNpPfFmMkK"))
            except ValueError:
                continue
            if abs(v_prev_f) < 1e-9 and abs(v_curr) > 1e-9:
                mult = {"f": 1e-3, "p": 1.0, "n": 1e3, "u": 1e6, "m": 1e9}
                if t_str and t_str[-1].lower() in mult:
                    times.append(float(t_str[:-1]) * mult[t_str[-1].lower()])
                else:
                    times.append(float(t_str) * 1e12)
    return sorted(times)


def _find_output_phase_col(headers: list[str]) -> Optional[str]:
    """Output is observed at the load: a phase column ``P(...LOADOUT...)``; else the
    last phase column."""
    for h in headers:
        if "LOADOUT" in h.upper() and h.upper().startswith("P("):
            return h
    phase_cols = [h for h in headers if h.startswith("P(")]
    return phase_cols[-1] if phase_cols else None


def parse_josim_csv(csv_path: Path, testbench_path: Optional[Path] = None) -> dict:
    """Parse a JoSIM output CSV into metrics: per-junction SFQ pulse counts (winding
    number), propagation delay, a functional flag, and peak current."""
    import numpy as np
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with open(csv_path) as f:
        header = f.readline().strip().replace('"', '').split(",")
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    cols = {h: data[:, i] for i, h in enumerate(header)}
    phase_cols = [h for h in header if h.startswith("P(")]
    current_cols = [h for h in header if h.startswith("I(")]
    metrics = {"sim_duration_ns": float((t[-1] - t[0]) * 1e9)}

    PULSE_THRESH = 1.5 * np.pi
    pulse_counts, pulse_times = {}, {}
    for pc in phase_cols:
        phi = cols[pc]
        pulse_counts[pc] = int(round((phi[-1] - phi[0]) / (2 * np.pi)))
        crossings, ref = [], phi[0]
        for i in range(1, len(phi)):
            if phi[i] - ref >= PULSE_THRESH:
                crossings.append(float(t[i] * 1e12)); ref = phi[i]
        pulse_times[pc] = crossings
    metrics["pulse_counts"] = pulse_counts

    out_pc = _find_output_phase_col(header)
    if testbench_path and Path(testbench_path).exists():
        in_times = _extract_input_pulse_times(Path(testbench_path).read_text())
        if out_pc and in_times and pulse_times.get(out_pc):
            out_times = pulse_times[out_pc]
            deltas = [ot - max([it for it in in_times if it < ot], default=ot)
                      for ot in out_times if any(it < ot for it in in_times)]
            if deltas:
                metrics["propagation_delay_ps"] = round(float(np.median(deltas)), 3)
            metrics["n_input_pulses"] = len(in_times)
            metrics["n_output_pulses"] = len(out_times)
            metrics["functional_match"] = len(out_times) > 0
        elif out_pc and pulse_times.get(out_pc):
            metrics["n_output_pulses"] = len(pulse_times[out_pc])
            metrics["functional_match"] = len(pulse_times[out_pc]) > 0
        else:
            metrics["functional_match"] = False
    elif len(phase_cols) >= 2:
        in_pc, out_pc = phase_cols[0], phase_cols[-1]
        in_t, out_t = pulse_times.get(in_pc, []), pulse_times.get(out_pc, [])
        if in_t and out_t:
            metrics["propagation_delay_ps"] = round(out_t[0] - in_t[0], 3)
        metrics["functional_match"] = (pulse_counts.get(in_pc, 0) > 0 and
            pulse_counts.get(out_pc, 0) >= pulse_counts.get(in_pc, 0) - 1)

    if current_cols:
        metrics["peak_current_uA"] = round(
            max(float(np.abs(cols[c]).max()) for c in current_cols) * 1e6, 3)
    return metrics
