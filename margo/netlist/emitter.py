"""
emitter.py — NetlistGraph -> JoSIM-runnable ``.cir`` text.

Faithfulness contract: an element whose ``value`` was *not* changed re-emits its
original raw token (so untouched parts of the deck are stable); an element a
generator modified (``modified=True``) re-formats its value via
``units.format_value``. Opaque lines (comments, controls, .model/.param raw) emit
verbatim, preserving the deck's structure and JoSIM semantics.
"""
from __future__ import annotations

from . import units
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


def _value_token(el: Element) -> str:
    """Return the value token for an element: raw if untouched, else formatted."""
    if el.modified and el.value is not None:
        return units.format_value(el.value)
    if el.raw is not None:
        return el.raw
    if el.value is not None:
        return units.format_value(el.value)
    return ""


def _params_str(params: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def _emit_element(el: Element) -> str:
    if isinstance(el, JJ):
        parts = [el.name, *el.nodes, el.model]
        if el.params:
            parts.append(_params_str(el.params))
        return " ".join(parts)

    if isinstance(el, (Inductor, Resistor, Capacitor)):
        parts = [el.name, *el.nodes, _value_token(el)]
        if el.params:
            parts.append(_params_str(el.params))
        return " ".join(p for p in parts if p)

    if isinstance(el, Source):
        parts = [el.name, *el.nodes]
        if el.func is not None:
            parts.append(el.func)
        else:
            parts.append(_value_token(el))
        return " ".join(p for p in parts if p)

    if isinstance(el, TLine):
        parts = [el.name, *el.nodes]
        if el.params:
            parts.append(_params_str(el.params))
        return " ".join(parts)

    if isinstance(el, Mutual):
        # nodes hold inductor names; value is the coupling coefficient.
        return " ".join([el.name, *el.nodes, _value_token(el)])

    if isinstance(el, SubcktInstance):
        if el.subckt_first:
            parts = [el.name, el.subckt, *el.nodes]
        else:
            parts = [el.name, *el.nodes, el.subckt]
        if el.params:
            parts.append(_params_str(el.params))
        return " ".join(parts)

    # Fallback for any future element type.
    return " ".join([el.name, *el.nodes, _value_token(el)]).rstrip()


def _emit_scope(g: NetlistGraph, lines: list[str]) -> None:
    for item in g.items:
        if isinstance(item, RawLine):
            lines.append(item.text)
        elif isinstance(item, Model):
            lines.append(item.text())
        elif isinstance(item, Param):
            lines.append(item.raw)
        elif isinstance(item, Subckt):
            lines.append(item.raw_header)
            _emit_scope(item.graph, lines)
            lines.append(item.raw_footer)
        elif isinstance(item, Element):
            lines.append(_emit_element(item))
        else:
            # Unknown item; skip silently rather than corrupt the deck.
            continue


def emit_cir(g: NetlistGraph) -> str:
    """Render a NetlistGraph back to ``.cir`` text."""
    lines: list[str] = []
    _emit_scope(g, lines)
    return "\n".join(lines) + "\n"
