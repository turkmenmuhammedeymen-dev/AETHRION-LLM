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
            "model_id": "code", "dataset_sha256": "a" * 64,
            "release_status": "NOT_RELEASED"
        }), encoding="utf-8")
        evaluation = folder / "evaluation_report.json"
        evaluation.write_text(json.dumps({
            "model_id": "code", "dataset_sha256": "a" * 64,
            "status": "approved_for_release"
        }), encoding="utf-8")
        card = folder / "MODEL_CARD.md"
        card.write_text("# AETHRION-CODE test fixture", encoding="utf-8")
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
