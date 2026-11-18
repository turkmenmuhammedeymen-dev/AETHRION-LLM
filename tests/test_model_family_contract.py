"""Cross-check the eleven-model manifest against native runtime architecture code.

These are registration/contract checks only. They do not load checkpoints,
train models, or assert model quality/release readiness.
"""
import importlib
import json
from pathlib import Path
import unittest

from aethrion.native_registry import NativeModelRegistry
from aethrion.routing import Capability


EXPECTED_ARCHITECTURES = {
    "chat": "aethrion.model:AETHRIONLM",
    "code": "aethrion.code_model:AETHRIONCodeLM",
    "vision": "aethrion.vision_model:AETHRIONVision",
    "image": "aethrion.image_model:AETHRIONImageGenerator",
    "asr": "aethrion.asr_model:AETHRIONASR",
    "tts": "aethrion.tts_model:AETHRIONTTS",
    "vocoder": "aethrion.tts_model:AETHRIONVocoder",
    "3d": "aethrion.model_3d:AETHRION3D",
    "research": "aethrion.research_model:AETHRIONResearch",
    "document": "aethrion.document_model:AETHRIONDocument",
    "safety": "aethrion.safety_model:AETHRIONSafe",
}

EXPECTED_ROUTABLE_CAPABILITIES = {
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety",
}


class ModelFamilyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest = json.loads(
            (cls.root / "models" / "manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_contains_exactly_the_eleven_native_models(self):
        models = self.manifest["models"]
        self.assertEqual(set(models), set(EXPECTED_ARCHITECTURES))
        self.assertEqual(len(models), 11)

    def test_manifest_architecture_targets_exist_in_source(self):
        for model_id, expected in EXPECTED_ARCHITECTURES.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(self.manifest["models"][model_id]["architecture"], expected)
                module_name, class_name = expected.split(":", 1)
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                self.assertTrue(callable(model_class))

    def test_registry_reports_all_eleven_as_not_trained(self):
        registry = NativeModelRegistry(self.root, "models/manifest.json")
        self.assertEqual(set(registry.capabilities()), set(EXPECTED_ARCHITECTURES))
        for model_id in EXPECTED_ARCHITECTURES:
            with self.subTest(model_id=model_id):
                self.assertEqual(registry.status(model_id), "not_trained")

    def test_user_routing_capabilities_exclude_internal_vocoder(self):
        self.assertEqual({cap.value for cap in Capability}, EXPECTED_ROUTABLE_CAPABILITIES)
        self.assertNotIn("vocoder", {cap.value for cap in Capability})
        self.assertIn("vocoder", self.manifest["models"])


if __name__ == "__main__":
    unittest.main()
