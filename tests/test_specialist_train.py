import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aethrion.specialist_train import train_specialist


class SpecialistTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path, approved: bool):
        source = root / "source.txt"
        source.write_text("Synthetic test-only code text.", encoding="utf-8")
        dataset = root / "dataset.npz"
        np.savez(
            dataset,
            sample_id=np.asarray(["s1", "s2", "s3", "s4", "s5", "s6"]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            token_ids=np.random.default_rng(42).integers(0, 256, size=(6, 12), dtype=np.int64),
        )
        manifest = root / "manifest.json"
        payload = {
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "synthetic-unit-test-only",
            "model_id": "code",
            "modality": "code",
            "training_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "dataset_path": "dataset.npz",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "synthetic-fixture",
                "source_uri": "unit-test://synthetic",
                "source_path": "source.txt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "Synthetic unit test fixture",
                "permitted_use": "This fixture may only be used by the unit test",
                "evidence_uri": "unit-test://fixture-license",
                "allow_training": approved,
                "privacy_reviewed": approved,
            }],
        }
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    def _config(self, root: Path):
        return {
            "model_id": "code",
            "model_config": {
                "vocab_size": 256, "block_size": 16, "n_layer": 1,
                "n_head": 2, "n_embd": 16, "dropout": 0.0
            },
            "output_dir": str(root / "artifacts"),
            "seed": 11,
            "max_steps": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "save_interval": 1,
        }

    def test_unapproved_specialist_data_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=False)
            with self.assertRaisesRegex(ValueError, "Training blocked"):
                train_specialist(self._config(root), manifest)
            self.assertFalse((root / "artifacts" / "checkpoint_last.pt").exists())

    def test_code_specialist_training_emits_integrity_checked_unreleased_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            report = train_specialist(self._config(root), manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())
            self.assertEqual(report["release_status"], "NOT_RELEASED")
            self.assertTrue((root / "artifacts" / "metrics.jsonl").is_file())
            self.assertTrue((root / "artifacts" / "training_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
