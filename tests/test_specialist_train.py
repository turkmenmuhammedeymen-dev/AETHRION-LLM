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
                "license": "CC0-1.0",
                "permitted_use": "Unit-test training fixture only; not production training data",
                "evidence_uri": "https://example.invalid/fixture-license",
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

    def test_specialist_artifact_paths_reject_escape_and_allow_local_paths(self):
        from aethrion.specialist_train import _resolve_artifact_path
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder).resolve()
            for invalid in ("../outside", str(Path(outside) / "artifacts")):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "escapes|parent traversal"):
                        _resolve_artifact_path(root, invalid, "output_dir")
            self.assertEqual(_resolve_artifact_path(root, "artifacts/models/code", "output_dir"), root / "artifacts/models/code")

    def test_specialist_artifact_paths_reject_symlink_escape(self):
        from aethrion.specialist_train import _resolve_artifact_path
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder).resolve()
            (root / "linked-artifacts").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                _resolve_artifact_path(root, "linked-artifacts/run", "output_dir")

    def test_specialist_training_rejects_external_output_before_data_access(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            config = self._config(root)
            config["output_dir"] = str(Path(outside) / "artifacts")
            with self.assertRaisesRegex(ValueError, "escapes"):
                train_specialist(config, root / "missing-manifest.json")

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
                "format": "aethrion-specialist-checkpoint-v2",
                "model_id": "code", "step": 0, "model_config": {},
                "model_state": {}, "optimizer_state": {},
                "dataset_sha256": "0" * 64, "tokenizer_sha256": "1" * 64,
                "manifest_sha256": "2" * 64, "rng_state": {},
                "training_config": {
                    "seed": 11, "batch_size": 2, "learning_rate": 0.001,
                    "weight_decay": 0.0, "grad_clip": 1.0,
                    "eval_interval": 1, "save_interval": 1, "max_steps": 2,
                },
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

    def test_training_resume_rejects_changed_optimizer_or_batch_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            config["max_steps"] = 2
            train_specialist(config, manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"
            changed = dict(config)
            changed["batch_size"] = 3
            with self.assertRaisesRegex(ValueError, "training settings changed"):
                train_specialist(changed, manifest, resume=checkpoint)

    def test_training_resume_rejects_shorter_original_training_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            config["max_steps"] = 2
            train_specialist(config, manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"
            shortened = dict(config)
            shortened["max_steps"] = 1
            with self.assertRaisesRegex(ValueError, "max_steps cannot be lower"):
                train_specialist(shortened, manifest, resume=checkpoint)

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


    def test_resume_rejects_changed_source_and_approval_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, approved=True)
            config = self._config(root)
            config["max_steps"] = 2
            train_specialist(config, manifest)
            checkpoint = root / "artifacts" / "checkpoint_last.pt"

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["sources"][0]["permitted_use"] = (
                "Changed after checkpoint creation; resume must be rejected."
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest changed"):
                train_specialist(config, manifest, resume=checkpoint)

    def test_specialist_checkpoint_recovers_when_latest_rng_snapshot_is_malformed(self):
        from aethrion.specialist_train import _capture_rng_state, _load_checkpoint, _save_checkpoint, _sha256

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_last.pt"
            payload = {
                "format": "aethrion-specialist-checkpoint-v2",
                "model_id": "code", "step": 0, "model_config": {},
                "model_state": {}, "optimizer_state": {},
                "dataset_sha256": "0" * 64, "tokenizer_sha256": "1" * 64,
                "manifest_sha256": "2" * 64, "rng_state": _capture_rng_state(),
                "training_config": {
                    "seed": 11, "batch_size": 2, "learning_rate": 0.001,
                    "weight_decay": 0.0, "grad_clip": 1.0,
                    "eval_interval": 1, "save_interval": 1, "max_steps": 2,
                },
            }
            _save_checkpoint(checkpoint, payload)
            _save_checkpoint(checkpoint, {**payload, "step": 1})
            torch.save({**payload, "step": 2, "rng_state": {}}, checkpoint)
            Path(str(checkpoint) + ".sha256").write_text(_sha256(checkpoint) + "\\n", encoding="ascii")

            with self.assertWarnsRegex(RuntimeWarning, "recovering from"):
                recovered = _load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(recovered["step"], 0)
            self.assertEqual(recovered["model_id"], "code")

    def test_checkpoint_with_invalid_checksum_and_no_backup_fails_closed(self):
        from aethrion.specialist_train import _load_checkpoint, _save_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_last.pt"
            payload = {
                "format": "aethrion-specialist-checkpoint-v2",
                "model_id": "code", "step": 0, "model_config": {},
                "model_state": {}, "optimizer_state": {},
                "dataset_sha256": "0" * 64, "tokenizer_sha256": "1" * 64,
                "manifest_sha256": "2" * 64, "rng_state": {},
                "training_config": {
                    "seed": 11, "batch_size": 2, "learning_rate": 0.001,
                    "weight_decay": 0.0, "grad_clip": 1.0,
                    "eval_interval": 1, "save_interval": 1, "max_steps": 2,
                },
            }
            _save_checkpoint(checkpoint, payload)
            Path(str(checkpoint) + ".sha256").write_text("f" * 64 + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "integrity verification failed"):
                _load_checkpoint(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
