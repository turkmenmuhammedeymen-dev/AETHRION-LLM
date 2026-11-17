import json
from pathlib import Path
import tempfile
import unittest

from aethrion.family_audit import (
    EXPECTED_MODELS,
    FamilyAuditError,
    audit_model_family,
)


class FamilyAuditTests(unittest.TestCase):
    def _repo(self, root: Path):
        (root / "models").mkdir()
        (root / "configs" / "specialists").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        (root / "data" / "specialists").mkdir(parents=True)
        models = {}
        for model_id in EXPECTED_MODELS:
            if model_id == "chat":
                config = "configs/small.json"
                dataset = "data/processed/corpus_manifest.json"
            else:
                config = f"configs/specialists/{model_id}.json"
                dataset = f"data/specialists/manifests/{model_id}.json"
                (root / config).write_text('{"width": 8}', encoding="utf-8")
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
            self.assertFalse(report["training_started"])
            self.assertFalse(report["automatic_approval"])
            self.assertEqual(
                {row["model_id"] for row in report["models"]},
                set(EXPECTED_MODELS),
            )

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
