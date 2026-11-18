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
            group_id=np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in ("ga", "gb", "gc", "gd", "ge", "gf")]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            token_ids=np.random.default_rng(1).integers(0, 256, size=(6, 12), dtype=np.int64),
        )
        from aethrion.tokenizer import ByteBPETokenizer
        tokenizer_path = root / "tokenizer.json"
        ByteBPETokenizer().save(tokenizer_path)
        payload = {
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "unit-test-only",
            "model_id": "code",
            "modality": "code",
            "training_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "governance": {
                "schema": "aethrion-human-review-v1",
                "model_id": "code",
                "approval_status": "approved",
                "author_role": "unit-test author",
                "authorship_attested": True,
                "first_party_origin_attested": True,
                "rights_holder_role": "unit-test rights holder",
                "rights_attested": True,
                "source_rights_reviewed": True,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "reviewer_role": "unit-test reviewer",
                "reviewed_at": "2026-01-01T00:00:00Z",
                "evidence_uri": "unit-test://human-review",
            },
            "dataset_path": "dataset.npz",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "tokenizer_path": "tokenizer.json",
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "synthetic-fixture",
                "source_uri": "unit-test://synthetic-fixture",
                "source_path": "source.txt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "SYNTHETIC_UNIT_TEST_FIXTURE",
                "permitted_use": "Unit tests only; no production training",
                "evidence_uri": "unit-test://license-fixture",
                "allow_training": approved,
                "privacy_reviewed": approved,
            }],
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, payload

    def test_training_array_integrity_rejects_non_finite_float_values(self):
        from aethrion.specialist_data import _validate_array_integrity
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SpecialistDataError, "NaN or infinite"):
                    _validate_array_integrity({"images": np.asarray([[value]], dtype=np.float32)}, 1)

    def test_training_array_integrity_rejects_misaligned_and_non_numeric_arrays(self):
        from aethrion.specialist_data import _validate_array_integrity
        with self.assertRaisesRegex(SpecialistDataError, "axis 0"):
            _validate_array_integrity({"features": np.zeros((2, 4), dtype=np.float32)}, 3)
        with self.assertRaisesRegex(SpecialistDataError, "numeric or boolean"):
            _validate_array_integrity({"features": np.asarray([["bad"]])}, 1)

    def test_unapproved_dataset_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory), approved=False)
            with self.assertRaisesRegex(SpecialistDataError, "Training blocked"):
                validate_specialist_dataset(manifest)

    def test_approved_test_fixture_is_schema_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory), approved=True)
            result = validate_specialist_dataset(manifest, require_training_approval=False)
            self.assertEqual(result["model_id"], "code")
            self.assertEqual(result["sample_count"], 6)
            self.assertEqual(result["split_counts"], {"train": 2, "validation": 2, "test": 2})

    def test_pending_source_license_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload = self._fixture(Path(directory), approved=True)
            payload["sources"][0]["license"] = "PENDING_HUMAN_REVIEW"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "license is pending, unknown, or test-only"):
                validate_specialist_dataset(manifest)

    def test_evaluation_only_source_permission_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload = self._fixture(Path(directory), approved=True)
            payload["sources"][0]["license"] = "CC0-1.0"
            payload["sources"][0]["permitted_use"] = "Evaluation only"
            payload["sources"][0]["evidence_uri"] = "https://example.invalid/license-fixture"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "does not explicitly authorize model training"):
                validate_specialist_dataset(manifest)

    def test_negative_training_permission_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload = self._fixture(Path(directory), approved=True)
            payload["sources"][0]["license"] = "CC0-1.0"
            payload["sources"][0]["permitted_use"] = "Not for training; evaluation only"
            payload["sources"][0]["evidence_uri"] = "https://example.invalid/license-fixture"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "excludes model training"):
                validate_specialist_dataset(manifest)

    def test_non_https_source_evidence_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload = self._fixture(Path(directory), approved=True)
            payload["sources"][0]["license"] = "CC0-1.0"
            payload["sources"][0]["permitted_use"] = "AI model training and evaluation"
            payload["sources"][0]["evidence_uri"] = "http://example.invalid/license"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "evidence must be an HTTPS URL"):
                validate_specialist_dataset(manifest)

    def test_duplicate_specialist_source_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            payload["sources"].append(dict(payload["sources"][0], source_path="source.txt"))
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "Duplicate specialist source_id"):
                validate_specialist_dataset(manifest)

    def test_duplicate_specialist_source_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            payload["sources"].append(dict(payload["sources"][0], source_id="second-source"))
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "Duplicate specialist source_path"):
                validate_specialist_dataset(manifest)

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

    def test_group_id_cannot_cross_data_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            dataset = root / "dataset.npz"
            with np.load(dataset, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
            shared_group = hashlib.sha256(b"shared-provenance-group").hexdigest()
            arrays["group_id"][0] = shared_group
            arrays["group_id"][2] = shared_group
            np.savez(dataset, **arrays)
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "crosses"):
                validate_specialist_dataset(manifest)

    def test_exact_duplicate_code_inputs_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            dataset = root / "dataset.npz"
            with np.load(dataset, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
            arrays["token_ids"][2] = arrays["token_ids"][0]
            np.savez(dataset, **arrays)
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "duplicate token_ids input.*train and validation"):
                validate_specialist_dataset(manifest)

    def test_duplicate_chat_prompts_are_detected_even_when_responses_differ(self):
        from aethrion.specialist_data import _validate_input_split_uniqueness
        arrays = {
            "input_ids": np.asarray([[10, 11, 20, 21], [10, 11, 30, 31]], dtype=np.int64),
            "labels": np.asarray([[-100, -100, 20, 21], [-100, -100, 30, 31]], dtype=np.int64),
        }
        with self.assertRaisesRegex(SpecialistDataError, "duplicate prompt_ids input"):
            _validate_input_split_uniqueness("chat", arrays, ["train", "test"])

    def test_asr_duplicate_inputs_ignore_different_padding_tails(self):
        from aethrion.specialist_data import _validate_input_split_uniqueness
        arrays = {
            "features": np.asarray([
                [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]],
                [[0.1, 0.2], [0.3, 0.4], [9.0, 9.0]],
            ], dtype=np.float32),
            "input_lengths": np.asarray([2, 2], dtype=np.int64),
        }
        with self.assertRaisesRegex(SpecialistDataError, "duplicate features input"):
            _validate_input_split_uniqueness("asr", arrays, ["train", "validation"])

    def test_shared_vision_prompt_is_not_mistaken_for_duplicate_image(self):
        from aethrion.specialist_data import _validate_input_split_uniqueness
        arrays = {
            "images": np.asarray([
                np.zeros((3, 16, 16), dtype=np.uint8),
                np.ones((3, 16, 16), dtype=np.uint8),
            ]),
            "token_ids": np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64),
        }
        _validate_input_split_uniqueness("vision", arrays, ["train", "test"])
        arrays["images"][1] = arrays["images"][0]
        with self.assertRaisesRegex(SpecialistDataError, "duplicate images input"):
            _validate_input_split_uniqueness("vision", arrays, ["train", "test"])

    def test_tokenizer_hash_mismatch_blocks_specialist_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, approved=True)
            (root / "tokenizer.json").write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "tokenizer SHA-256 mismatch"):
                validate_specialist_dataset(manifest)

    def test_token_ids_outside_verified_tokenizer_vocabulary_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            dataset = root / "dataset.npz"
            with np.load(dataset, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
            arrays["token_ids"][0, 0] = 256
            np.savez(dataset, **arrays)
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "outside the verified tokenizer vocabulary"):
                validate_specialist_dataset(manifest)

    def _safety_fixture(self, root: Path, per_class: int):
        source = root / "safety-source.txt"
        source.write_text("Synthetic taxonomy-only safety fixture.", encoding="utf-8")
        sample_ids = []
        splits = []
        decisions = []
        for split in ("train", "validation", "test"):
            for label in range(3):
                for index in range(per_class):
                    sample_ids.append(f"{split}-{label}-{index}")
                    splits.append(split)
                    decisions.append(label)
        count = len(sample_ids)
        dataset = root / "safety.npz"
        reasons = np.zeros((count, 16), dtype=np.int64)
        reasons[:, 0] = 1
        np.savez(
            dataset,
            sample_id=np.asarray(sample_ids),
            group_id=np.asarray([hashlib.sha256(value.encode()).hexdigest() for value in sample_ids]),
            split=np.asarray(splits),
            token_ids=np.random.default_rng(7).integers(0, 256, size=(count, 12), dtype=np.int64),
            decision_labels=np.asarray(decisions, dtype=np.int64),
            reason_labels=reasons,
        )
        from aethrion.tokenizer import ByteBPETokenizer
        tokenizer_path = root / "tokenizer.json"
        ByteBPETokenizer().save(tokenizer_path)
        payload = {
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "synthetic-safety-taxonomy-test-only",
            "model_id": "safety",
            "modality": "text",
            "training_allowed": True,
            "owner_approval": True,
            "privacy_reviewed": True,
            "governance": {
                "schema": "aethrion-human-review-v1",
                "model_id": "safety",
                "approval_status": "approved",
                "author_role": "unit-test author",
                "authorship_attested": True,
                "first_party_origin_attested": True,
                "rights_holder_role": "unit-test rights holder",
                "rights_attested": True,
                "source_rights_reviewed": True,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "reviewer_role": "unit-test reviewer",
                "reviewed_at": "2026-01-01T00:00:00Z",
                "evidence_uri": "unit-test://human-review",
                "safety_reviewed": True,
            },
            "safety_min_per_class": 25,
            "dataset_path": "safety.npz",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "tokenizer_path": "tokenizer.json",
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "synthetic-taxonomy-fixture",
                "source_uri": "unit-test://safety-taxonomy",
                "source_path": "safety-source.txt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "SYNTHETIC_UNIT_TEST_FIXTURE",
                "permitted_use": "Unit tests only; no production training",
                "evidence_uri": "unit-test://safety-fixture",
                "allow_training": True,
                "privacy_reviewed": True,
            }],
        }
        manifest = root / "safety-manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    def test_safety_dataset_requires_minimum_per_decision_in_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._safety_fixture(Path(directory), per_class=25)
            result = validate_specialist_dataset(manifest, require_training_approval=False)
            self.assertEqual(result["model_id"], "safety")
            self.assertEqual(result["split_counts"], {"train": 75, "validation": 75, "test": 75})

    def test_safety_dataset_blocks_underrepresented_decision_class(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._safety_fixture(Path(directory), per_class=24)
            with self.assertRaisesRegex(SpecialistDataError, "at least 25"):
                validate_specialist_dataset(manifest)

    def test_missing_held_out_split_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, payload = self._fixture(root, approved=True)
            dataset = root / "dataset.npz"
            np.savez(
                dataset,
                sample_id=np.asarray(["a", "b", "c"]),
                group_id=np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in ("ga", "gb", "gc")]),
                split=np.asarray(["train", "train", "validation"]),
                token_ids=np.random.default_rng(1).integers(0, 256, size=(3, 12), dtype=np.int64),
            )
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SpecialistDataError, "train, validation and test"):
                validate_specialist_dataset(manifest)


if __name__ == "__main__":
    unittest.main()
