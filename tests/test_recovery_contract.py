"""Focused recovery-contract tests using temporary, non-production fixtures.

These tests never write production artifacts or alter model training statuses.
"""
import json
import signal
import tempfile
import unittest
from pathlib import Path

import torch

from aethrion import train
from aethrion.checkpoint import load_checkpoint, save_checkpoint
from aethrion import specialist_train


class ContinuousTrainingRecoveryContractTests(unittest.TestCase):
    def test_base_checkpoint_round_trip_and_corruption_rejection(self):
        payload = {
            "format_version": 4,
            "step": 7,
            "model": {},
            "optimizer": {},
            "scaler": {},
            "model_config": {},
            "data_sha256": "0" * 64,
            "tokenizer_sha256": "1" * 64,
            "rng_state": {},
            "training_config": {"stage": "temporary-recovery-test"},
        }
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "base.pt"
            digest = save_checkpoint(payload, checkpoint)
            self.assertEqual(len(digest), 64)
            self.assertEqual(load_checkpoint(checkpoint)["step"], 7)
            checkpoint.write_bytes(b"corrupted temporary checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
                load_checkpoint(checkpoint)

    def test_specialist_checkpoint_round_trip_and_corruption_rejection(self):
        payload = {
            "format": "aethrion-specialist-checkpoint-v2",
            "model_id": "code",
            "step": 3,
            "model_config": {},
            "model_state": {},
            "optimizer_state": {},
            "dataset_sha256": "0" * 64,
            "tokenizer_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "rng_state": {},
            "training_config": {
                "seed": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "eval_interval": 1,
                "save_interval": 1,
                "max_steps": 4,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "specialist.pt"
            specialist_train._save_checkpoint(checkpoint, payload)
            loaded = specialist_train._load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(loaded["step"], 3)
            checkpoint.write_bytes(b"corrupted temporary specialist checkpoint")
            with self.assertRaisesRegex(ValueError, "integrity verification failed"):
                specialist_train._load_checkpoint(checkpoint, torch.device("cpu"))

    def test_sigterm_handlers_request_checkpointed_stop(self):
        train._STOP_REQUESTED = False
        specialist_train._STOP_REQUESTED = False
        try:
            train._request_stop(signal.SIGTERM, None)
            specialist_train._request_stop(signal.SIGTERM, None)
            self.assertTrue(train._STOP_REQUESTED)
            self.assertTrue(specialist_train._STOP_REQUESTED)
        finally:
            train._STOP_REQUESTED = False
            specialist_train._STOP_REQUESTED = False

    def test_all_eleven_production_model_statuses_remain_not_trained(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["models"]), 11)
        self.assertEqual(
            {entry.get("status") for entry in manifest["models"].values()},
            {"not_trained"},
        )


    def test_real_sigterm_delivery_in_child_process_saves_checkpoint(self):
        import os
        import subprocess
        import sys
        import time

        with tempfile.TemporaryDirectory() as folder:
            root = Path(__file__).resolve().parents[1]
            ready = Path(folder) / "ready"
            checkpoint = Path(folder) / "signal-stop.pt"
            child_code = """
import signal
import sys
import time
from pathlib import Path
from aethrion import train
from aethrion.checkpoint import save_checkpoint
ready, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
train._STOP_REQUESTED = False
signal.signal(signal.SIGTERM, train._request_stop)
ready.write_text("ready", encoding="utf-8")
while not train._STOP_REQUESTED:
    time.sleep(0.01)
save_checkpoint({
    "format_version": 4, "step": 11, "model": {}, "optimizer": {},
    "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
    "tokenizer_sha256": "1" * 64, "rng_state": {},
    "training_config": {"stage": "isolated-signal-test"},
}, checkpoint)
"""
            process = subprocess.Popen(
                [sys.executable, "-c", child_code, str(ready), str(checkpoint)],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "child process did not install its SIGTERM handler")
                os.kill(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, msg=f"stdout={stdout} stderr={stderr}")
                self.assertEqual(load_checkpoint(checkpoint)["step"], 11)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
