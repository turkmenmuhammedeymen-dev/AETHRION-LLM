import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aethrion.native_registry import (
    NativeModelError,
    NativeModelIntegrityError,
    NativeModelRegistry,
    NativeModelUnavailable,
)


def write_manifest(root: Path, entry: dict, *, capability: str = "chat") -> Path:
    models = {} if entry is None else {capability: entry}
    path = root / "models" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "format": "aethrion-native-model-family-v1",
        "execution_policy": "local_only",
        "models": models,
    }), encoding="utf-8")
    return path


class NativeModelRegistryTests(unittest.TestCase):
    def test_untrained_model_is_not_resolvable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "models/chat.pt",
                "status": "not_trained",
            })
            registry = NativeModelRegistry(root)
            self.assertEqual(registry.status("chat"), "not_trained")
            with self.assertRaises(NativeModelUnavailable):
                registry.resolve("chat")

    def test_valid_local_artifacts_resolve_after_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            artifacts = {
                "checkpoint.pt": b"first-party-checkpoint",
                "tokenizer.json": b'{"format":"aethrion-test"}',
                "training.json": json.dumps({
                    "model_id": "chat", "dataset_sha256": "a" * 64,
                    "status": "instruction_tuning_complete_not_released",
                    "release_status": "NOT_RELEASED",
                    "checkpoint_sha256": hashlib.sha256(b"first-party-checkpoint").hexdigest()
                }).encode(),
                "review.txt": b"Human release review evidence for a synthetic unit test fixture.",
                "evaluation.json": json.dumps({
                    "model_id": "chat", "dataset_sha256": "a" * 64,
                    "checkpoint_sha256": hashlib.sha256(b"first-party-checkpoint").hexdigest(),
                    "status": "approved_for_release",
                    "quality_gate_passed": True,
                    "safety_gate_passed": True,
                    "human_reviewed": True,
                    "reviewer": "unit-test-reviewer",
                    "reviewed_at": "2026-11-17T15:00:00Z",
                    "approval_evidence_path": "models/review.txt",
                    "approval_evidence_sha256": hashlib.sha256(
                        b"Human release review evidence for a synthetic unit test fixture."
                    ).hexdigest(),
                    "benchmarks": {
                        "held_out_lm_loss": {"metric": "loss", "value": 1.0, "threshold": 2.0, "passed": True, "evidence": "held-out corpus report"},
                        "instruction_following": {"metric": "pass_rate", "value": 0.9, "threshold": 0.8, "passed": True, "evidence": "human-reviewed evaluation"}
                    }
                }).encode(),
                "MODEL_CARD.md": (
                    b"# Test-only model card\\n\\n## Intended use\\nTest only\\n"
                    b"## Training data\\nSynthetic only\\n## Evaluation\\nUnit tests\\n"
                    b"## Limitations\\nNot a real model\\n"
                ),
            }
            digests = {}
            for name, raw in artifacts.items():
                target = models / name
                target.write_bytes(raw)
                digests[name] = hashlib.sha256(raw).hexdigest()
            entry = {
                "architecture": "aethrion.model:AETHRIONLM",
                "model_config": {"vocab_size": 256},
                "checkpoint": "models/checkpoint.pt",
                "checkpoint_sha256": digests["checkpoint.pt"],
                "tokenizer": "models/tokenizer.json",
                "tokenizer_sha256": digests["tokenizer.json"],
                "training_report": "models/training.json",
                "training_report_sha256": digests["training.json"],
                "evaluation_report": "models/evaluation.json",
                "evaluation_report_sha256": digests["evaluation.json"],
                "model_card": "models/MODEL_CARD.md",
                "model_card_sha256": digests["MODEL_CARD.md"],
                "dataset_sha256": "a" * 64,
                "release_approval": True,
                "quality_gate_passed": True,
                "safety_gate_passed": True,
                "status": "trained",
            }
            write_manifest(root, entry)
            registry = NativeModelRegistry(root)
            resolved = registry.resolve("chat")
            self.assertEqual(resolved.checkpoint, (models / "checkpoint.pt").resolve())
            self.assertEqual(resolved.tokenizer, (models / "tokenizer.json").resolve())
            self.assertEqual(resolved.model_config, {"vocab_size": 256})
            self.assertEqual(registry.status("chat"), "ready")

    def test_trained_model_without_release_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(NativeModelError, "release_approval=true"):
                write_manifest(root, {
                    "architecture": "aethrion.model:AETHRIONLM",
                    "model_config": {"vocab_size": 256},
                    "checkpoint": "models/chat.pt",
                    "checkpoint_sha256": "0" * 64,
                    "tokenizer": "models/tokenizer.json",
                    "tokenizer_sha256": "0" * 64,
                    "training_report": "models/train.json",
                    "training_report_sha256": "0" * 64,
                    "evaluation_report": "models/eval.json",
                    "evaluation_report_sha256": "0" * 64,
                    "model_card": "models/card.md",
                    "model_card_sha256": "0" * 64,
                    "dataset_sha256": "0" * 64,
                    "release_approval": False,
                    "quality_gate_passed": True,
                    "safety_gate_passed": True,
                    "status": "trained",
                })
                NativeModelRegistry(root)
    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            checkpoint = models / "chat.pt"
            checkpoint.write_bytes(b"tampered")
            tokenizer = models / "tokenizer.json"
            tokenizer.write_text('{"format":"test"}', encoding="utf-8")
            training = models / "training.json"
            training.write_text(json.dumps({
                "model_id": "chat", "dataset_sha256": "a" * 64
            }), encoding="utf-8")
            evaluation = models / "evaluation.json"
            evaluation.write_text(json.dumps({
                "model_id": "chat", "dataset_sha256": "a" * 64,
                "status": "approved_for_release"
            }), encoding="utf-8")
            card = models / "MODEL_CARD.md"
            card.write_text("# Test model", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "model_config": {"vocab_size": 256},
                "checkpoint": "models/chat.pt",
                "checkpoint_sha256": "0" * 64,
                "tokenizer": "models/tokenizer.json",
                "tokenizer_sha256": digest(tokenizer),
                "training_report": "models/training.json",
                "training_report_sha256": digest(training),
                "evaluation_report": "models/evaluation.json",
                "evaluation_report_sha256": digest(evaluation),
                "model_card": "models/MODEL_CARD.md",
                "model_card_sha256": digest(card),
                "dataset_sha256": "a" * 64,
                "release_approval": True,
                "quality_gate_passed": True,
                "safety_gate_passed": True,
                "status": "trained",
            })
            registry = NativeModelRegistry(root)
            self.assertEqual(registry.status("chat"), "integrity_failed")
            with self.assertRaises(NativeModelIntegrityError):
                registry.resolve("chat")

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "../outside.pt",
                "checkpoint_sha256": "0" * 64,
                "status": "trained",
            })
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)

    def test_remote_url_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "https://example.invalid/model.pt",
                "checkpoint_sha256": "0" * 64,
                "status": "trained",
            })
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)

    def test_unknown_capability_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "models" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "format": "aethrion-native-model-family-v1",
                "execution_policy": "local_only",
                "models": {"unknown": {}},
            }), encoding="utf-8")
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)


if __name__ == "__main__":
    unittest.main()
