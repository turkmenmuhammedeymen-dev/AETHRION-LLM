import json
from pathlib import Path
import tempfile
import unittest

from aethrion.product_integration import (
    ProductRouteError,
    load_product_routes,
    product_route_model_readiness,
    validate_product_routes,
)


class FakeRegistry:
    def __init__(self, states):
        self.states = states

    def status(self, model_id):
        return self.states.get(model_id, "not_trained")


class ProductIntegrationTests(unittest.TestCase):
    def test_repository_route_contract_loads_and_is_native_only(self):
        payload = load_product_routes()
        self.assertFalse(payload["external_fallback"])
        self.assertEqual(payload["execution_policy"], "local_only")
        self.assertGreaterEqual(len(payload["routes"]), 10)

    def test_external_fallback_is_rejected(self):
        payload = load_product_routes()
        payload["external_fallback"] = True
        with self.assertRaises(ProductRouteError):
            validate_product_routes(payload)

    def test_unknown_model_is_rejected(self):
        payload = load_product_routes()
        payload["routes"][0]["required_models"] = ["unknown-model"]
        with self.assertRaises(ProductRouteError):
            validate_product_routes(payload)

    def test_duplicate_route_ids_are_rejected(self):
        payload = load_product_routes()
        payload["routes"][1]["route_id"] = payload["routes"][0]["route_id"]
        with self.assertRaises(ProductRouteError):
            validate_product_routes(payload)

    def test_untrained_model_keeps_product_route_unavailable(self):
        result = product_route_model_readiness(
            FakeRegistry({"chat": "not_trained"}), "aethrion_chat"
        )
        self.assertEqual(result["status"], "models_unavailable")
        self.assertFalse(result["external_fallback"])
        self.assertFalse(result["endpoint_integration_verified"])

    def test_model_ready_does_not_claim_endpoint_integration(self):
        result = product_route_model_readiness(
            FakeRegistry({"chat": "ready"}), "aethrion_chat"
        )
        self.assertEqual(result["status"], "models_ready_adapter_unverified")
        self.assertFalse(result["endpoint_integration_verified"])

    def test_non_ai_assessment_request_is_not_misrepresented_as_scanner(self):
        result = product_route_model_readiness(
            FakeRegistry({}), "sentrix_assessment_request"
        )
        self.assertEqual(result["status"], "non_ai_request_only")
        self.assertEqual(result["required_models"], {})


if __name__ == "__main__":
    unittest.main()
