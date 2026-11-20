"""Durability guarantees for checkpoint and progress-file publication."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethrion import checkpoint


class CheckpointDurabilityTests(unittest.TestCase):
    def _payload(self, step):
        return {
            "format_version": 4,
            "step": step,
            "model": {},
            "optimizer": {},
            "scaler": {},
            "model_config": {},
            "data_sha256": "0" * 64,
            "tokenizer_sha256": "1" * 64,
            "rng_state": {},
            "training_config": {"stage": "durability-test"},
        }

    def test_checkpoint_fsyncs_directory_after_publishing_generations(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "checkpoint.pt"
            with mock.patch(
                "aethrion.checkpoint._fsync_directory",
                wraps=checkpoint._fsync_directory,
            ) as sync_directory:
                checkpoint.save_checkpoint(self._payload(0), target)
                checkpoint.save_checkpoint(self._payload(1), target)

            self.assertGreaterEqual(sync_directory.call_count, 3)
            self.assertEqual(checkpoint.verify_checkpoint(target), checkpoint.sha256_file(target))
            self.assertEqual(checkpoint.verify_checkpoint(Path(str(target) + ".previous")),
                             checkpoint.sha256_file(Path(str(target) + ".previous")))
            self.assertFalse(Path(str(target) + ".tmp").exists())
            self.assertFalse(Path(str(target) + ".sha256.tmp").exists())

    def test_atomic_progress_json_is_flushed_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "progress.json"
            with mock.patch(
                "aethrion.checkpoint._fsync_directory",
                wraps=checkpoint._fsync_directory,
            ) as sync_directory:
                checkpoint.write_json_atomic({"step": 17, "status": "interrupted"}, target)

            self.assertEqual(checkpoint.json.loads(target.read_text(encoding="utf-8"))
                             if hasattr(checkpoint, "json") else
                             __import__("json").loads(target.read_text(encoding="utf-8")),
                             {"step": 17, "status": "interrupted"})
            sync_directory.assert_called_once_with(target.parent)
            self.assertFalse(Path(str(target) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
