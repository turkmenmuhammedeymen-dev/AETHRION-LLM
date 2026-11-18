"""Strict preflight validation for numeric model and optimizer settings."""
from __future__ import annotations

import math
from typing import Any


def require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    """Require an exact integer (booleans are deliberately not accepted)."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
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
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
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
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must contain only finite numeric values.")
