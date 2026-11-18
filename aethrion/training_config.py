"""Strict preflight validation for numeric model and optimizer settings."""
from __future__ import annotations

import math
from typing import Any


def require_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Require an exact bounded integer (booleans are deliberately not accepted)."""
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {bounds}.")
    return value


def require_finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Require a finite int/float within the supplied bounds."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and (
        number < minimum if minimum_inclusive else number <= minimum
    ):
        operator = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}.")
    if maximum is not None and (
        number > maximum if maximum_inclusive else number >= maximum
    ):
        operator = "<=" if maximum_inclusive else "<"
        raise ValueError(f"{name} must be {operator} {maximum}.")
    return number


def validate_numeric_tree(value: Any, name: str = "model_config") -> None:
    """Require a JSON tree of finite numeric hyperparameters."""
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{name} must not be empty.")
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} contains an invalid field name.")
            validate_numeric_tree(child, f"{name}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_numeric_tree(child, f"{name}[{index}]")
        return
    require_finite_number(value, name)
