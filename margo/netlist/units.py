"""
units.py — SPICE/JoSIM engineering value parsing and formatting.

JoSIM (like SPICE) writes numeric values with an optional engineering prefix
(p, n, u, m, k, meg, ...) and an optional, *ignored* physical unit suffix
(V, A, F, H, S, Hz, Ohm). Examples seen in real RSFQ netlists:

    2.031p      -> 2.031e-12      (inductance, pH)
    0.07pF      -> 0.07e-12       (capacitance; F suffix ignored)
    2.8mV       -> 2.8e-3         (voltage; V suffix ignored)
    0.1mA       -> 0.1e-3         (current; A suffix ignored)
    5.23        -> 5.23           (plain)
    10M         -> 10e-3          (M == milli in SPICE, case-insensitive)
    0.693       -> 0.693          (coupling coefficient, unitless)

Note the SPICE convention: a single 'm'/'M' is *milli*; mega is 'meg'.
"""
from __future__ import annotations

import re

# Engineering prefixes. 'meg' must be tried before single-letter 'm'.
_PREFIX = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}

# number, optional prefix, optional (ignored) physical unit.
_VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"   # 1: number
    r"\s*(meg|[fpnumkgt])?"                              # 2: engineering prefix
    r"\s*(ohm|hz|[vafhs])?\s*$",                         # 3: physical unit (ignored)
    re.IGNORECASE,
)


def parse_value(token: str) -> float:
    """Parse a SPICE/JoSIM numeric token to a float in SI base units.

    Raises ValueError if the token is not a plain numeric value (e.g. it is a
    source function like ``pwl(...)`` — those are handled by the parser, not
    here).
    """
    m = _VALUE_RE.match(token)
    if not m:
        raise ValueError(f"not a numeric value token: {token!r}")
    number = float(m.group(1))
    prefix = m.group(2)
    if prefix:
        number *= _PREFIX[prefix.lower()]
    return number


def is_value(token: str) -> bool:
    return _VALUE_RE.match(token) is not None


# Prefixes preferred when formatting, largest-magnitude first.
_FMT_PREFIXES = [
    ("t", 1e12), ("g", 1e9), ("meg", 1e6), ("k", 1e3),
    ("", 1.0),
    ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15),
]


def format_value(value: float) -> str:
    """Format a float as a compact engineering token JoSIM can read.

    Chooses a prefix so the mantissa lands in [1, 1000) where possible. Used
    only when a generator *modifies* a value; unmodified elements re-emit their
    original token verbatim to guarantee a faithful round-trip.
    """
    if value == 0:
        return "0"
    av = abs(value)
    for suffix, scale in _FMT_PREFIXES:
        m = av / scale
        if 1.0 <= m < 1000.0:
            mant = value / scale
            # trim trailing zeros, keep it readable
            s = f"{mant:.6g}"
            return f"{s}{suffix}"
    # Fallback to scientific notation for extreme magnitudes.
    return f"{value:.6g}"
