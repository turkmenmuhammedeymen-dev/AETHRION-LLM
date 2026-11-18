import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import aethrion.specialist_train as specialist_train
from aethrion.specialist_train import _to_batch, train_specialist


class SpecialistTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path, approved: bool):
        source = root / "source.txt"
        source.write_text("Synthetic test-only code text.", encoding="utf-8")
        dataset = root / "dataset.npz"
        np.savez(
            dataset,
            sample_id=np.asarray(["s1", "s2", "s3", "s4", "s5", "s6"]),
            group_id=np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in ("g1", "g2", "g3", "g4", "g5", "g6")]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            token_ids=np.random.default_rng(42).integers(0, 256, size=(6, 12), dtype=np.int64),
        )
        from aethrion.tokenizer import ByteBPETokenizer
        tokenizer_path = root / "tokenizer.json"
        ByteBPETokenizer().save(tokenizer_path)
        manifest = root / "manifest.json"
        payload = {
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "synthetic-unit-test-only",
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

    def test_invalid_numeric_specialist_config_fails_before_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._config(root)
            invalid_cases = [
                ("learning_rate", float("nan")),
                ("learning_rate", float("inf")),
                ("weight_decay", float("nan")),
                ("grad_clip", float("inf")),
                ("max_steps", True),
                ("model_config", {**base["model_config"], "dropout": float("nan")}),
                ("seed", 2**32),
                ("model_config", {**base["model_config"], "width": 10**1000}),
            ]
            missing_manifest = root / "not-created-manifest.json"
            for key, value in invalid_cases:
                candidate = dict(base)
                candidate[key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        train_specialist(candidate, missing_manifest)

    def test_unapproved_specialist_data_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=False)
            with self.assertRaisesRegex(ValueError, "Training blocked"):
                train_specialist(self._config(root), manifest)
            self.assertFalse((root / "artifacts" / "checkpoint_last.pt").exists())

    def test_safety_reason_labels_are_converted_to_integer_tensors(self):
        arrays = {"reason_labels": np.asarray([[0, 2]], dtype=np.int32)}
        batch = _to_batch(arrays, ["reason_labels"], torch.tensor([0]), torch.device("cpu"))
        self.assertEqual(batch["reason_labels"].dtype, torch.long)

    def test_signal_stop_writes_checkpoint_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            config["max_steps"] = 2
            specialist_train._STOP_REQUESTED = False
            original_loss = specialist_train._loss

            def request_stop_after_training_batch(model, model_id, batch):
                loss = original_loss(model, model_id, batch)
                if model.training:
                    specialist_train._STOP_REQUESTED = True
                return loss

            try:
                with patch("aethrion.specialist_train._loss", side_effect=request_stop_after_training_batch):
                    paused = train_specialist(config, manifest)
                self.assertEqual(paused["status"], "training_interrupted_not_released")
                self.assertEqual(paused["steps"], 1)
                checkpoint = root / "artifacts" / "checkpoint_last.pt"
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())

                specialist_train._STOP_REQUESTED = False
                resumed = train_specialist(config, manifest, resume=checkpoint)
                self.assertEqual(resumed["status"], "training_complete_not_released")
                self.assertEqual(resumed["steps"], 2)
            finally:
                specialist_train._STOP_REQUESTED = False

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


    def test_specialist_checkpoint_recovers_when_checksum_valid_latest_is_unsafe(self):
        import warnings
        from aethrion.specialist_train import _load_checkpoint, _save_checkpoint

        class UnsafePayload:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_last.pt"
            base = {
                "format": "aethrion-specialist-checkpoint-v1",
                "model_id": "code", "step": 0, "model_config": {},
                "model_state": {}, "optimizer_state": {},
                "dataset_sha256": "0" * 64, "tokenizer_sha256": "1" * 64,
                "manifest_sha256": "2" * 64, "rng_state": {},
            }
            _save_checkpoint(checkpoint, base)
            _save_checkpoint(checkpoint, {**base, "step": 1})
            unsafe = {**base, "step": 2, "extra": UnsafePayload()}
            torch.save(unsafe, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            Path(str(checkpoint) + ".sha256").write_text(digest + "\n", encoding="ascii")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = _load_checkpoint(checkpoint, torch.device("cpu"))

            self.assertEqual(recovered["step"], 0)
            self.assertTrue(any("safely loaded" in str(item.message) for item in caught))

    def test_direct_training_call_clears_stale_stop_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            specialist_train._STOP_REQUESTED = True
            report = train_specialist(self._config(root), manifest)
            self.assertEqual(report["status"], "training_complete_not_released")
            self.assertFalse(specialist_train._STOP_REQUESTED)

    def test_training_resumes_and_preserves_verified_previous_checkpoint(self):
        from aethrion.specialist_train import _load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            config["max_steps"] = 2
            report = train_specialist(config, manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"
            previous = Path(str(checkpoint) + ".previous")
            self.assertTrue(previous.is_file())
            state = _load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(state["step"], 1)
            checkpoint.write_bytes(b"corrupt newest generation")
            recovered = _load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(recovered["step"], 0)
            self.assertEqual(recovered["model_id"], "code")
            self.assertEqual(report["release_status"], "NOT_RELEASED")

    def test_training_resume_continues_after_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            first = train_specialist(config, manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"
            config["max_steps"] = 2
            second = train_specialist(config, manifest, resume=checkpoint)
            self.assertEqual(second["steps"], 2)
            self.assertEqual(second["dataset_sha256"], first["dataset_sha256"])
            self.assertEqual(second["release_status"], "NOT_RELEASED")


if __name__ == "__main__":
    unittest.main()
