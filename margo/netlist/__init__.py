"""
netlist — topology-faithful, lossless RSFQ netlist representation.

A real circuit graph that round-trips to JoSIM losslessly, so verification of novel
topologies is trustworthy by construction. Parse a ``.cir`` deck to a NetlistGraph,
edit it in place, and emit it back.
"""
from __future__ import annotations

from . import units
from .emitter import emit_cir
from .graph import (
    Capacitor,
    Element,
    Inductor,
    JJ,
    Model,
    Mutual,
    NetlistGraph,
    Param,
    RawLine,
    Resistor,
    Source,
    Subckt,
    SubcktInstance,
    TLine,
)
from .parser import parse_cir

__all__ = [
    "units",
    "parse_cir",
    "emit_cir",
    "NetlistGraph",
    "Element",
    "JJ",
    "Inductor",
    "Resistor",
    "Capacitor",
    "Source",
    "TLine",
    "Mutual",
    "SubcktInstance",
    "Model",
    "Param",
    "Subckt",
    "RawLine",
]
