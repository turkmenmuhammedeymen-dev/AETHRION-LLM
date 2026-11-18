import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict

import torch

from aethrion.code_model import AETHRIONCodeLM, CodeModelConfig
from aethrion.native_registry import NativeModelUnavailable
from aethrion.native_runtime import NativeModelRuntime
from aethrion.tokenizer import ByteBPETokenizer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeModelRuntimeTests(unittest.TestCase):
    def _released_code_model(self, root: Path):
        folder = root / "models"
        folder.mkdir()
        config = CodeModelConfig(
            vocab_size=256, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
        )
        model = AETHRIONCodeLM(config)
        checkpoint = folder / "code.pt"
        torch.save({
            "format": "aethrion-specialist-checkpoint-v1",
            "model_id": "code",
            "step": 3,
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "dataset_sha256": "a" * 64,
        }, checkpoint)
        tokenizer = ByteBPETokenizer()
        tokenizer.train("AETHRION native code model fixture. " * 8, vocab_size=256)
        tokenizer_path = folder / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        training = folder / "training_report.json"
        training.write_text(json.dumps({
            "status": "training_complete_not_released",
            "model_id": "code", "dataset_sha256": "a" * 64,
            "checkpoint_sha256": digest(checkpoint),
            "release_status": "NOT_RELEASED"
        }), encoding="utf-8")
        review = folder / "release_review.txt"
        review.write_text("Synthetic runtime unit test review evidence.", encoding="utf-8")
        evaluation = folder / "evaluation_report.json"
        evaluation.write_text(json.dumps({
            "model_id": "code", "dataset_sha256": "a" * 64,
            "status": "approved_for_release",
            "checkpoint_sha256": digest(checkpoint),
            "quality_gate_passed": True, "safety_gate_passed": True,
            "human_reviewed": True, "reviewer": "unit-test-reviewer",
            "reviewed_at": "2026-11-17T16:00:00Z",
            "approval_evidence_path": "models/release_review.txt",
            "approval_evidence_sha256": digest(review),
            "benchmarks": {
                "unit_test_pass_rate": {"metric": "pass_rate", "direction": "max", "value": 1.0, "threshold": 0.95, "passed": True, "evidence": "Synthetic test fixture"},
                "security_regression": {"metric": "regression_count", "direction": "min", "value": 0.0, "threshold": 0.0, "passed": True, "evidence": "No regression in synthetic test fixture"}
            }
        }), encoding="utf-8")
        card = folder / "MODEL_CARD.md"
        card.write_text(
            "# AETHRION-CODE test fixture\\n\\n"
            "## Intended use\\nSynthetic tests\\n"
            "## Training data\\nSynthetic fixture\\n"
            "## Evaluation\\nUnit tests only\\n"
            "## Limitations\\nNo real code quality is claimed\\n",
            encoding="utf-8"
        )
        return {
            "architecture": "aethrion.code_model:AETHRIONCodeLM",
            "model_config": asdict(config),
            "checkpoint": "models/code.pt",
            "checkpoint_sha256": digest(checkpoint),
            "tokenizer": "models/tokenizer.json",
            "tokenizer_sha256": digest(tokenizer_path),
            "training_report": "models/training_report.json",
            "training_report_sha256": digest(training),
            "evaluation_report": "models/evaluation_report.json",
            "evaluation_report_sha256": digest(evaluation),
            "model_card": "models/MODEL_CARD.md",
            "model_card_sha256": digest(card),
            "dataset_sha256": "a" * 64,
            "release_approval": True,
            "quality_gate_passed": True,
            "safety_gate_passed": True,
            "status": "trained",
        }

    def test_released_local_code_checkpoint_loads_and_generates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self._released_code_model(root)
            manifest_dir = root / "models"
            manifest_dir.mkdir(exist_ok=True)
            (manifest_dir / "manifest.json").write_text(json.dumps({
                "format": "aethrion-native-model-family-v1",
                "execution_policy": "local_only",
                "models": {"code": entry},
            }), encoding="utf-8")
            runtime = NativeModelRuntime(root)
            result = runtime.generate_text("code", "write a function", max_new_tokens=2)
            self.assertEqual(result["model"], "AETHRION-CODE")
            self.assertIsInstance(result["text"], str)
            self.assertEqual(runtime.status()["code"], "ready")

    def _untrained_runtime(self, root: Path) -> NativeModelRuntime:
        folder = root / "models"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps({
            "format": "aethrion-native-model-family-v1",
            "execution_policy": "local_only",
            "models": {
                "chat": {
                    "architecture": "aethrion.model:AETHRIONLM",
                    "status": "not_trained",
                },
                "code": {
                    "architecture": "aethrion.code_model:AETHRIONCodeLM",
                    "status": "not_trained",
                },
            },
        }), encoding="utf-8")
        return NativeModelRuntime(root)

    def test_generate_text_rejects_invalid_sampling_parameters_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self._untrained_runtime(Path(directory))
            invalid = [
                {"max_new_tokens": True},
                {"max_new_tokens": 1.5},
                {"temperature": float("nan")},
                {"temperature": float("inf")},
                {"temperature": 0},
                {"top_k": True},
                {"top_k": 0},
                {"top_k": 1.5},
                {"top_p": float("nan")},
                {"top_p": 0},
                {"top_p": 1.1},
            ]
            for options in invalid:
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        runtime.generate_text("code", "prompt", **options)
            self.assertEqual(runtime._models, {})

    def test_generate_text_rejects_empty_prompt_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self._untrained_runtime(Path(directory))
            with self.assertRaises(ValueError):
                runtime.generate_text("chat", "   ")
            self.assertEqual(runtime._models, {})

    def test_untrained_model_is_not_loaded_or_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "models"
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({
                "format": "aethrion-native-model-family-v1",
                "execution_policy": "local_only",
                "models": {"code": {
                    "architecture": "aethrion.code_model:AETHRIONCodeLM",
                    "model_config": asdict(CodeModelConfig(
                        vocab_size=256, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0
                    )),
                    "checkpoint": "models/code.pt",
                    "tokenizer": "models/tokenizer.json",
                    "status": "not_trained",
                }},
            }), encoding="utf-8")
            runtime = NativeModelRuntime(root)
            self.assertEqual(runtime.status()["code"], "not_trained")
            with self.assertRaises(NativeModelUnavailable):
                runtime.model("code")


if __name__ == "__main__":
    unittest.main()
