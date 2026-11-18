"""Repository-level checks for the ten specialist dataset registrations.

These manifests are pending registrations, not datasets or training approvals.
"""
import json
from pathlib import Path
import unittest

MODELS = {
    "code": "code",
    "vision": "vision",
    "image": "image",
    "asr": "audio",
    "tts": "audio",
    "vocoder": "audio",
    "3d": "3d",
    "research": "text",
    "document": "document",
    "safety": "text",
}


class SpecialistManifestRegistryTests(unittest.TestCase):
    def test_all_ten_specialists_have_well_formed_manifests(self):
        root = Path(__file__).resolve().parents[1]
        for model_id, modality in MODELS.items():
            with self.subTest(model_id=model_id):
                path = root / "data" / "specialists" / "manifests" / f"{model_id}.json"
                self.assertTrue(path.is_file(), f"Missing specialist manifest: {model_id}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload.get("format"), "aethrion-specialist-dataset-v1")
                self.assertEqual(payload.get("model_id"), model_id)
                self.assertEqual(payload.get("modality"), modality)

    def test_pending_manifests_cannot_authorize_training(self):
        root = Path(__file__).resolve().parents[1]
        for model_id in MODELS:
            with self.subTest(model_id=model_id):
                path = root / "data" / "specialists" / "manifests" / f"{model_id}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIs(payload.get("training_allowed"), False)
                self.assertIs(payload.get("owner_approval"), False)
                self.assertIs(payload.get("privacy_reviewed"), False)
                self.assertEqual(payload.get("governance", {}).get("approval_status"), "pending")
                self.assertEqual(payload.get("sources"), [])
                self.assertEqual(payload.get("dataset_status"),
                                 "awaiting_owner_authored_data_and_independent_review")


if __name__ == "__main__":
    unittest.main()
