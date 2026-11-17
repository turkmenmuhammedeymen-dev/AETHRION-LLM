import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aethrion.specialist_data import (
    SpecialistDataError,
    validate_specialist_dataset,
)


class SpecialistDataApprovalTests(unittest.TestCase):
    def _fixture(self, root: Path, *, approved=True):
        source = root / "source.txt"
        source.write_text("Synthetic test-only code corpus.", encoding="utf-8")
        dataset = root / "dataset.npz"
        np.savez(
            dataset,
            sample_id=np.asarray(["a", "b", "c", "d", "e", "f"]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            token_ids=np.random.default_rng(1).integers(0, 300, size=(6, 12), dtype=np.int64),
        )
        payload = {
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "unit-test-only",
            "model_id": "code",
            "modality": "code",
            "training_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "dataset_path": "dataset.npz",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "synthetic-fixture",
                "source_uri": "unit-test://synthetic-fixture",
                "source_path": "source.txt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "Synthetic unit-test fixture",
                "permitted_use": "Unit tests only; not a production training corpus",
                "evidence_uri": "unit-test://license-fixture",
                "allow_training": approved,
                "privacy_reviewed": approved,
            }],
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, payload

    def test_unapproved_dataset_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory), approved=False)
            with self.assertRaisesRegex(SpecialistDataError, "Training blocked"):
                validate_specialist_dataset(manifest)

    def test_approved_test_fixture_is_schema_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory), approved=True)
            result = validate_specialist_dataset(manifest)
            self.assertEqual(result["model_id"], "code")
            self.assertEqual(result["sample_count"], 6)
            self.assertEqual(result["split_counts"], {"train": 2, "validation": 2, "test": 2})

    def test_source_hash_mismatch_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            (root / "source.txt").write_text("changed source content", encoding="utf-8")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "source file SHA-256 mismatch"):
                validate_specialist_dataset(manifest)

    def test_dataset_hash_mismatch_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            payload["dataset_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "dataset SHA-256 mismatch"):
                validate_specialist_dataset(manifest)

    def test_missing_held_out_split_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            dataset = root / "dataset.npz"
            np.savez(
                dataset,
                sample_id=np.asarray(["a", "b", "c"]),
                split=np.asarray(["train", "train", "validation"]),
                token_ids=np.random.default_rng(1).integers(0, 300, size=(3, 12), dtype=np.int64),
            )
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "train, validation and test"):
                validate_specialist_dataset(manifest)


if __name__ == "__main__":
    unittest.main()
