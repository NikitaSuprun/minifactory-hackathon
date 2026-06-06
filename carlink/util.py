"""Small value-coercion helpers. The board sends some values as strings
("true"/"false"/"255"); these normalize them without throwing."""

from __future__ import annotations

from typing import Any

_TRUE = {"true", "1", "on", "yes"}
_FALSE = {"false", "0", "off", "no"}


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
