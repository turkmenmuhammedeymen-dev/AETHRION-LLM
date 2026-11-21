"""Durability guarantees for checkpoint and progress-file publication."""
import json
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
            previous = Path(str(target) + ".previous")
            self.assertEqual(checkpoint.verify_checkpoint(previous), checkpoint.sha256_file(previous))
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

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"step": 17, "status": "interrupted"},
            )
            sync_directory.assert_called_once_with(target.parent)
            self.assertFalse(Path(str(target) + ".tmp").exists())


    def test_resume_reconciles_metrics_to_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.jsonl"
            target.write_text(
                json.dumps({"step": 0, "loss": 1.0}) + "\n"
                + json.dumps({"step": 1, "loss": 0.9}) + "\n"
                + json.dumps({"step": 2, "loss": 0.8}) + "\n"
                + json.dumps({"step": 1, "loss": 0.7}) + "\n"
                + "{truncated json",
                encoding="utf-8",
            )

            discarded = checkpoint.reconcile_metrics_jsonl(target, checkpoint_step=1)

            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(discarded, 3)
            self.assertEqual([record["step"] for record in records], [0, 1])
            self.assertFalse(Path(str(target) + ".tmp").exists())

    def test_resume_metrics_reconciliation_rejects_invalid_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.jsonl"
            target.write_text('{"step": 0}\n', encoding="utf-8")
            for invalid in (True, -1, 1.5):
                with self.assertRaisesRegex(ValueError, "checkpoint_step"):
                    checkpoint.reconcile_metrics_jsonl(target, invalid)


    def test_resume_metrics_reconciliation_restores_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.jsonl"
            target.write_text('{"step": 0, "loss": 1.0}', encoding="utf-8")

            discarded = checkpoint.reconcile_metrics_jsonl(target, checkpoint_step=0)

            self.assertEqual(discarded, 0)
            self.assertTrue(target.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"step": 0, "loss": 1.0})


if __name__ == "__main__":
    unittest.main()
