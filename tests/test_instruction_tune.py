import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from aethrion.checkpoint import save_checkpoint, load_checkpoint, sha256_file
import aethrion.instruction_tune as instruction_tune
from aethrion.instruction_tune import tune_instructions
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer


class InstructionTuningTests(unittest.TestCase):
    def _fixture(self, root: Path, approved=True):
        raw = root / "source.txt"
        raw.write_text("Synthetic instruction fixture, for unit tests only.", encoding="utf-8")
        tokenizer_path = root / "tokenizer.json"
        tokenizer = ByteBPETokenizer()
        tokenizer.train("Merhaba AETHRION.\nYanıt metni.\n" * 10, vocab_size=256)
        tokenizer.save(tokenizer_path)
        model_config = ModelConfig(
            vocab_size=256, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
        )
        base = AETHRIONLM(model_config)
        optimizer = torch.optim.AdamW(base.parameters(), lr=0.001)
        base_checkpoint = root / "base.pt"
        payload = {
            "format_version": 3,
            "step": 0,
            "model": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": {},
            "model_config": model_config.__dict__,
            "data_sha256": "d" * 64,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        save_checkpoint(payload, base_checkpoint)
        dataset = root / "instructions.npz"
        inputs = np.random.default_rng(4).integers(0, 256, size=(6, 16), dtype=np.int64)
        labels = np.full((6, 16), -100, dtype=np.int64)
        labels[:, 8:] = inputs[:, 8:]
        np.savez(
            dataset,
            sample_id=np.asarray(["a", "b", "c", "d", "e", "f"]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            input_ids=inputs,
            labels=labels,
        )
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "synthetic-instruction-test",
            "model_id": "chat",
            "modality": "text",
            "training_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "dataset_path": "instructions.npz",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "fixture",
                "source_uri": "unit-test://fixture",
                "source_path": "source.txt",
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "license": "Synthetic unit-test content",
                "permitted_use": "Unit tests only",
                "evidence_uri": "unit-test://evidence",
                "allow_training": approved,
                "privacy_reviewed": approved,
            }],
        }), encoding="utf-8")
        config = {
            "model_config": model_config.__dict__,
            "tokenizer_path": str(tokenizer_path),
            "output_dir": str(root / "instruction_output"),
            "seed": 42,
            "max_steps": 1,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "save_interval": 1,
        }
        return manifest, config, base_checkpoint

    def test_unapproved_instruction_data_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, config, base = self._fixture(Path(directory), approved=False)
            with self.assertRaisesRegex(ValueError, "Training blocked"):
                tune_instructions(config, manifest, base)
            self.assertFalse((Path(directory) / "instruction_output" / "checkpoint_last.pt").exists())

    def test_response_only_instruction_tuning_saves_unreleased_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, config, base = self._fixture(Path(directory), approved=True)
            report = tune_instructions(config, manifest, base)
            checkpoint = Path(config["output_dir"]) / "checkpoint_last.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())
            self.assertEqual(report["release_status"], "NOT_RELEASED")
            state = load_checkpoint(checkpoint)
            self.assertEqual(state["training_stage"], "instruction_tuning")

    def test_instruction_tuning_signal_pause_saves_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config, base = self._fixture(root, approved=True)
            config["max_steps"] = 2
            instruction_tune._STOP_REQUESTED = False
            original_step = torch.optim.AdamW.step

            def step_then_request_stop(optimizer, *args, **kwargs):
                result = original_step(optimizer, *args, **kwargs)
                instruction_tune._STOP_REQUESTED = True
                return result

            try:
                with patch.object(torch.optim.AdamW, "step", new=step_then_request_stop):
                    paused = tune_instructions(config, manifest, base)
                self.assertEqual(paused["status"], "instruction_tuning_interrupted_not_released")
                self.assertEqual(paused["steps"], 1)
                checkpoint = Path(config["output_dir"]) / "checkpoint_last.pt"
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())

                instruction_tune._STOP_REQUESTED = False
                resumed = tune_instructions(config, manifest, base, resume=checkpoint)
                self.assertEqual(resumed["status"], "instruction_tuning_complete_not_released")
                self.assertEqual(resumed["steps"], 2)
            finally:
                instruction_tune._STOP_REQUESTED = False


if __name__ == "__main__":
    unittest.main()
