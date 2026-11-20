import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from aethrion.artifact_semantics import audit_model_artifact_semantics
from aethrion.model import AETHRIONLM, ModelConfig


class ArtifactSemanticsTests(unittest.TestCase):
    def _fixture(self, root: Path):
        model_config = {
            "vocab_size": 256, "block_size": 8, "n_layer": 1,
            "n_head": 1, "n_embd": 8, "dropout": 0.0,
        }
        model = AETHRIONLM(ModelConfig(**model_config))
        checkpoint = root / "checkpoint.pt"
        torch.save({
            "format_version": 4,
            "step": 3,
            "model": model.state_dict(),
            "model_config": model_config,
            "data_sha256": "a" * 64,
        }, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        tokenizer = root / "tokenizer.json"
        tokenizer.write_text(json.dumps({"format": "aethrion-byte-bpe-v1", "merges": []}), encoding="utf-8")
        training = root / "training.json"
        training.write_text(json.dumps({
            "model_id": "chat", "step": 3, "dataset_sha256": "a" * 64,
            "checkpoint_sha256": digest, "release_status": "NOT_RELEASED",
            "status": "training_complete_not_released",
        }), encoding="utf-8")
        return model_config, checkpoint, digest, training, tokenizer

    def test_checkpoint_weights_and_training_report_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_config, checkpoint, digest, training, tokenizer = self._fixture(root)
            result = audit_model_artifact_semantics(root, "chat", {
                "model_config": model_config,
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": digest,
                "tokenizer": "tokenizer.json",
                "training_report": "training.json",
            })
            self.assertEqual(result["checkpoint"]["status"], "verified_structure")
            self.assertGreater(result["checkpoint"]["tensor_count"], 0)
            self.assertTrue(result["cross_artifact_consistent"])
            self.assertFalse(result["trained_claim_supported"])

    def test_checkpoint_weight_shape_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_config, checkpoint, digest, training, tokenizer = self._fixture(root)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            key = next(iter(payload["model"]))
            payload["model"][key] = torch.zeros((1,))
            torch.save(payload, checkpoint)
            result = audit_model_artifact_semantics(root, "chat", {
                "model_config": model_config, "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "tokenizer": "tokenizer.json",
                "training_report": "training.json",
            })
            self.assertEqual(result["checkpoint"]["status"], "weight_shape_mismatch")
            self.assertFalse(result["cross_artifact_consistent"])

    def test_report_checkpoint_hash_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_config, checkpoint, digest, training, tokenizer = self._fixture(root)
            report = json.loads(training.read_text(encoding="utf-8"))
            report["checkpoint_sha256"] = "f" * 64
            training.write_text(json.dumps(report), encoding="utf-8")
            result = audit_model_artifact_semantics(root, "chat", {
                "model_config": model_config, "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": digest, "tokenizer": "tokenizer.json",
                "training_report": "training.json",
            })
            self.assertIn("training_report_checkpoint_hash_mismatch",
                          result["consistency_problems"])
            self.assertFalse(result["cross_artifact_consistent"])

    def test_processor_wrong_model_identity_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processor = root / "processor.json"
            processor.write_text(json.dumps({"format": "test-v1", "model_id": "image"}), encoding="utf-8")
            result = audit_model_artifact_semantics(root, "vision", {"processor": "processor.json"})
            self.assertEqual(result["processor_status"], "model_id_mismatch")

    def test_unsafe_artifact_path_is_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = audit_model_artifact_semantics(root, "chat", {
                "checkpoint": "../outside.pt", "training_report": "../training.json",
            })
            self.assertEqual(result["checkpoint"]["status"], "missing")
            self.assertFalse(result["cross_artifact_consistent"])


if __name__ == "__main__":
    unittest.main()
