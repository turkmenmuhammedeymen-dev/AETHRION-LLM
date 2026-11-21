"""Fault-injection tests for durable checkpoint and JSON publication."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethrion import checkpoint as checkpoint_module
from aethrion.checkpoint import load_checkpoint, save_checkpoint, write_json_atomic


def _payload(step):
    return {
        "format_version": 4, "step": step, "model": {}, "optimizer": {},
        "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
        "tokenizer_sha256": "1" * 64, "rng_state": {},
        "training_config": {"stage": "fault-injection-test"},
    }


class CheckpointWriteFailureRecoveryTests(unittest.TestCase):
    def test_sidecar_replace_failure_recovers_previous_verified_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            save_checkpoint(_payload(1), target)
            real_replace = checkpoint_module.os.replace
            calls = {"count": 0}

            def fail_new_sidecar_once(source, destination):
                calls["count"] += 1
                if calls["count"] == 4:
                    raise OSError("injected sidecar publication failure")
                return real_replace(source, destination)

            with mock.patch.object(checkpoint_module.os, "replace", side_effect=fail_new_sidecar_once):
                with self.assertRaisesRegex(OSError, "injected sidecar"):
                    save_checkpoint(_payload(2), target)

            with self.assertWarnsRegex(RuntimeWarning, "recovering from"):
                recovered = load_checkpoint(target)
            self.assertEqual(recovered["step"], 1)
            for suffix in (".tmp", ".sha256.tmp", ".previous.tmp", ".previous.sha256.tmp"):
                self.assertFalse(Path(str(target) + suffix).exists())

    def test_checkpoint_replace_failure_preserves_latest_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            save_checkpoint(_payload(1), target)
            real_replace = checkpoint_module.os.replace
            calls = {"count": 0}

            def fail_checkpoint_once(source, destination):
                calls["count"] += 1
                if calls["count"] == 3:
                    raise OSError("injected checkpoint publication failure")
                return real_replace(source, destination)

            with mock.patch.object(checkpoint_module.os, "replace", side_effect=fail_checkpoint_once):
                with self.assertRaisesRegex(OSError, "injected checkpoint"):
                    save_checkpoint(_payload(2), target)

            self.assertEqual(load_checkpoint(target)["step"], 1)
            self.assertFalse(Path(str(target) + ".tmp").exists())
            self.assertFalse(Path(str(target) + ".sha256.tmp").exists())

    def test_json_replace_failure_preserves_old_report_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            write_json_atomic({"step": 1, "status": "running"}, target)
            with mock.patch.object(
                checkpoint_module.os, "replace",
                side_effect=OSError("injected JSON publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected JSON"):
                    write_json_atomic({"step": 2, "status": "running"}, target)

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             {"step": 1, "status": "running"})
            self.assertFalse(Path(str(target) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
