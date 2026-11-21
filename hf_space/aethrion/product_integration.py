"""Validation helpers for the cross-product AETHRION native-AI route contract.

This module validates capability declarations only. It does not execute tools,
perform inference, or claim a product integration is live.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aethrion.native_registry import CAPABILITIES


FORMAT = "aethrion-product-ai-routes-v1"
ALLOWED_ROUTE_STATUSES = {
    "blocked_until_native_release",
    "blocked_until_native_release_and_task_extension",
    "non_ai_request_only",
}


class ProductRouteError(ValueError):
    """Raised when a product route contract is invalid."""


def load_product_routes(path: str | Path = "configs/integrations/product_ai_routes.json") -> dict[str, Any]:
    route_path = Path(path)
    try:
        payload = json.loads(route_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductRouteError("Product AI route contract is missing or invalid JSON.") from exc
    validate_product_routes(payload)
    return payload


def validate_product_routes(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ProductRouteError("Unsupported product AI route contract format.")
    if payload.get("execution_policy") != "local_only" or payload.get("external_fallback") is not False:
        raise ProductRouteError("Product AI routes must enforce local_only execution with external fallback disabled.")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ProductRouteError("Product AI routes must be a non-empty list.")

    seen_ids: set[str] = set()
    for route in routes:
        if not isinstance(route, dict):
            raise ProductRouteError("Each product AI route must be an object.")
        route_id = route.get("route_id")
        if not isinstance(route_id, str) or not route_id.strip() or route_id in seen_ids:
            raise ProductRouteError("Product AI route IDs must be non-empty and unique.")
        seen_ids.add(route_id)
        if not isinstance(route.get("application"), str) or not route["application"].strip():
            raise ProductRouteError(f"{route_id} requires an application name.")
        if not isinstance(route.get("legacy_endpoint"), str) or not route["legacy_endpoint"].strip():
            raise ProductRouteError(f"{route_id} requires a legacy endpoint name.")
        if route.get("status") not in ALLOWED_ROUTE_STATUSES:
            raise ProductRouteError(f"{route_id} has an unsupported status.")
        required = route.get("required_models")
        optional = route.get("optional_models")
        if not isinstance(required, list) or not isinstance(optional, list):
            raise ProductRouteError(f"{route_id} model lists must be arrays.")
        if any(not isinstance(item, str) for item in required + optional):
            raise ProductRouteError(f"{route_id} model identifiers must be strings.")
        if len(set(required)) != len(required) or len(set(optional)) != len(optional):
            raise ProductRouteError(f"{route_id} contains duplicate model identifiers.")
        if set(required) & set(optional):
            raise ProductRouteError(f"{route_id} cannot require and optionally list the same model.")
        unknown = (set(required) | set(optional)) - CAPABILITIES
        if unknown:
            raise ProductRouteError(f"{route_id} references unknown native models: {', '.join(sorted(unknown))}.")
        if not isinstance(route.get("limitations"), list) or not all(
            isinstance(item, str) and item.strip() for item in route["limitations"]
        ):
            raise ProductRouteError(f"{route_id} must document its limitations.")
        if route["status"] != "non_ai_request_only" and not required:
            raise ProductRouteError(f"{route_id} must require at least one native model.")


def product_route_model_readiness(registry: Any, route_id: str,
                                  path: str | Path = "configs/integrations/product_ai_routes.json") -> dict[str, Any]:
    """Report model-artifact readiness without claiming endpoint integration readiness."""
    payload = load_product_routes(path)
    route = next((item for item in payload["routes"] if item["route_id"] == route_id), None)
    if route is None:
        raise ProductRouteError(f"Unknown product AI route: {route_id}")
    required = {model_id: registry.status(model_id) for model_id in route["required_models"]}
    optional = {model_id: registry.status(model_id) for model_id in route["optional_models"]}
    unavailable = {key: value for key, value in required.items() if value != "ready"}
    model_status = "models_unavailable" if unavailable else "models_ready_adapter_unverified"
    if route["status"] == "non_ai_request_only":
        model_status = "non_ai_request_only"
    return {
        "route_id": route_id,
        "application": route["application"],
        "legacy_endpoint": route["legacy_endpoint"],
        "status": model_status,
        "required_models": required,
        "optional_models": optional,
        "external_fallback": False,
        "endpoint_integration_verified": False,
    }
