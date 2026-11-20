import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aethrion.family_audit import (
    EXPECTED_MODELS,
    FamilyAuditError,
    audit_model_family,
)
from aethrion.specialist_data import MODEL_MODALITIES


class FamilyAuditTests(unittest.TestCase):
    def _repo(self, root: Path):
        (root / "models").mkdir()
        (root / "configs" / "specialists").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        (root / "data" / "specialists" / "manifests" / "templates").mkdir(parents=True)
        models = {}
        for model_id in EXPECTED_MODELS:
            if model_id == "chat":
                config = "configs/small.json"
            else:
                config = f"configs/specialists/{model_id}.json"
                (root / config).write_text('{"width": 8}', encoding="utf-8")
                template = {
                    "format": "aethrion-specialist-dataset-v1",
                    "model_id": model_id,
                    "modality": MODEL_MODALITIES[model_id],
                    "training_allowed": False,
                    "owner_approval": False,
                    "privacy_reviewed": False,
                    "sources": [],
                    "_template_notice": "Non-executable template; real reviewed data is required.",
                }
                if model_id in {"asr", "tts", "vocoder"}:
                    template["speaker_consent_reviewed"] = False
                template_path = (
                    root / "data" / "specialists" / "manifests" / "templates"
                    / f"{model_id}.template.json"
                )
                template_path.write_text(json.dumps(template), encoding="utf-8")
            models[model_id] = {
                "architecture": "aethrion.test:Model",
                "model_config": {"width": 8},
                "checkpoint": f"artifacts/{model_id}/checkpoint.pt",
                "status": "not_trained",
            }
        (root / "configs" / "small.json").write_text('{"width": 8}', encoding="utf-8")
        (root / "data" / "processed" / "corpus_manifest.json").write_text(
            '{"training_allowed": false}', encoding="utf-8"
        )
        (root / "data" / "sources.example.json").write_text(
            '{"training_allowed": false}', encoding="utf-8"
        )
        (root / "models" / "manifest.json").write_text(json.dumps({
            "format": "aethrion-native-model-family-v1",
            "execution_policy": "local_only",
            "models": models,
        }), encoding="utf-8")

    def test_reports_all_eleven_models_without_claiming_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            report = audit_model_family(root)
            self.assertEqual(report["model_count"], 11)
            self.assertEqual(report["trained_count"], 0)
            self.assertEqual(report["training_ready_count"], 0)
            self.assertEqual(report["release_ready_count"], 0)
            self.assertEqual(report["template_count"], 10)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["automatic_approval"])
            self.assertEqual(
                {row["model_id"] for row in report["models"]},
                set(EXPECTED_MODELS),
            )
            for row in report["models"]:
                self.assertFalse(row["training_preflight_passed"])
                if row["model_id"] != "chat":
                    self.assertFalse(row["dataset_manifest_present"])
                    self.assertTrue(row["template_manifest_present"])
                    self.assertTrue(row["template_manifest_valid"])
                    self.assertEqual(
                        row["template_manifest_status"], "non_executable_template"
                    )

    def test_template_with_approval_flag_is_not_reported_as_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            path = (
                root / "data" / "specialists" / "manifests" / "templates"
                / "code.template.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["training_allowed"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = audit_model_family(root)
            code = next(row for row in report["models"] if row["model_id"] == "code")
            self.assertFalse(code["template_manifest_valid"])
            self.assertEqual(code["template_manifest_status"], "invalid_template")
            self.assertEqual(report["template_count"], 9)
            self.assertFalse(code["training_ready"])


    def test_artifact_hash_is_verified_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            artifact = root / "artifacts" / "chat" / "checkpoint.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"checkpoint-bytes")
            digest = hashlib.sha256(b"checkpoint-bytes").hexdigest()
            manifest_path = root / "models" / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["models"]["chat"]["checkpoint_sha256"] = digest
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = audit_model_family(root)
            chat = next(row for row in report["models"] if row["model_id"] == "chat")
            self.assertEqual(chat["artifact_integrity"]["checkpoint"]["status"], "verified")
            self.assertEqual(chat["artifact_integrity"]["checkpoint"]["sha256"], digest)

            artifact.write_bytes(b"tampered-checkpoint")
            report = audit_model_family(root)
            chat = next(row for row in report["models"] if row["model_id"] == "chat")
            self.assertEqual(chat["artifact_integrity"]["checkpoint"]["status"], "hash_mismatch")
            self.assertFalse(chat["required_artifact_integrity_ok"])

    def test_present_artifact_without_registered_hash_is_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            artifact = root / "artifacts" / "chat" / "checkpoint.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"checkpoint-bytes")
            report = audit_model_family(root)
            chat = next(row for row in report["models"] if row["model_id"] == "chat")
            self.assertEqual(
                chat["artifact_integrity"]["checkpoint"]["status"], "hash_unregistered"
            )
            self.assertTrue(chat["checkpoint_present"])
            self.assertFalse(chat["required_artifact_integrity_ok"])

    def test_refuses_non_local_execution_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            path = root / "models" / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["execution_policy"] = "remote_fallback"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FamilyAuditError):
                audit_model_family(root)

    def test_refuses_missing_model_from_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            path = root / "models" / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].pop("safety")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FamilyAuditError):
                audit_model_family(root)


if __name__ == "__main__":
    unittest.main()
