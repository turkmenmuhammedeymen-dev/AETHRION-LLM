import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import torch
from aethrion.train import get_batch, capture_rng_state
from aethrion.tokenizer import ByteBPETokenizer
from aethrion.prepare_data import build_corpus
from aethrion.checkpoint import load_checkpoint

class TrainingUtilityTests(unittest.TestCase):
    def test_base_training_config_rejects_non_finite_and_inconsistent_values(self):
        from aethrion.train import validate_base_training_config

        valid = {
            "seed": 1337, "data_path": "corpus.txt", "tokenizer_path": "tokenizer.json",
            "output_dir": "artifacts", "vocab_size": 256, "block_size": 8,
            "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.1,
            "batch_size": 2, "gradient_accumulation_steps": 1,
            "learning_rate": 0.001, "min_learning_rate": 0.0001,
            "weight_decay": 0.0, "max_steps": 10, "warmup_steps": 2,
            "eval_interval": 1, "eval_batches": 1, "save_interval": 1,
            "grad_clip": 1.0, "amp": False,
        }
        validate_base_training_config(valid)
        invalid_cases = [
            ("learning_rate", float("nan")),
            ("learning_rate", float("inf")),
            ("min_learning_rate", float("nan")),
            ("weight_decay", float("inf")),
            ("grad_clip", float("nan")),
            ("dropout", float("nan")),
            ("dropout", 1.0),
            ("batch_size", True),
            ("amp", "false"),
            ("warmup_steps", 11),
            ("min_learning_rate", 0.002),
            ("seed", 2**32),
            ("n_embd", 10**1000),
        ]
        for key, value in invalid_cases:
            candidate = dict(valid)
            candidate[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_base_training_config(candidate)

    def test_checkpoint_rejects_malformed_sha256_sidecar(self):
        from aethrion.checkpoint import verify_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-bytes")
            Path(str(checkpoint) + ".sha256").write_text("not-a-sha256-value\\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
                verify_checkpoint(checkpoint)

    def test_checkpoint_sidecar_replace_crash_recovers_durable_previous_generation(self):
        from unittest.mock import patch
        import aethrion.checkpoint as checkpoint_module
        from aethrion.checkpoint import load_checkpoint, save_checkpoint
        from aethrion.train import capture_rng_state

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            base = {
                "format_version": 4, "step": 0, "model": {}, "optimizer": {},
                "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
                "tokenizer_sha256": "1" * 64, "rng_state": capture_rng_state(),
                "training_config": {"seed": 7},
            }
            save_checkpoint(base, checkpoint)
            original_replace = checkpoint_module.os.replace
            sidecar = Path(str(checkpoint) + ".sha256")

            def fail_live_sidecar(source, destination):
                if Path(destination) == sidecar:
                    raise OSError("simulated interruption before live sidecar promotion")
                return original_replace(source, destination)

            with patch("aethrion.checkpoint.os.replace", side_effect=fail_live_sidecar):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    save_checkpoint({**base, "step": 1}, checkpoint)

            # The live checkpoint/sidecar pair is split, but the previous pair
            # was made durable before either live file was replaced.
            recovered = load_checkpoint(checkpoint, map_location="cpu")
            self.assertEqual(recovered["step"], 0)
            previous = Path(str(checkpoint) + ".previous")
            self.assertEqual(
                checkpoint_module.verify_checkpoint(previous),
                checkpoint_module.sha256_file(previous),
            )

    def test_checkpoint_preserves_live_generation_at_each_precommit_crash_boundary(self):
        from unittest.mock import patch
        import aethrion.checkpoint as checkpoint_module
        from aethrion.checkpoint import load_checkpoint, save_checkpoint

        failure_targets = ("checkpoint.pt.previous", "checkpoint.pt.previous.sha256", "checkpoint.pt")
        for target_name in failure_targets:
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoint.pt"
                base = {
                    "format_version": 4, "step": 0, "model": {}, "optimizer": {},
                    "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
                    "tokenizer_sha256": "1" * 64, "rng_state": capture_rng_state(),
                    "training_config": {"seed": 7},
                }
                save_checkpoint(base, checkpoint)
                original_replace = checkpoint_module.os.replace
                target = Path(directory) / target_name

                def fail_at_boundary(source, destination):
                    if Path(destination) == target:
                        raise OSError("simulated checkpoint promotion interruption")
                    return original_replace(source, destination)

                with patch("aethrion.checkpoint.os.replace", side_effect=fail_at_boundary):
                    with self.assertRaisesRegex(OSError, "simulated checkpoint promotion"):
                        save_checkpoint({**base, "step": 1}, checkpoint)

                # Until the new live pair is committed, the old live generation
                # must remain readable. If backup promotion completed, it too
                # must be independently checksum-verifiable.
                self.assertEqual(load_checkpoint(checkpoint, map_location="cpu")["step"], 0)
                previous = Path(str(checkpoint) + ".previous")
                if previous.is_file():
                    self.assertEqual(
                        checkpoint_module.verify_checkpoint(previous),
                        checkpoint_module.sha256_file(previous),
                    )

    def test_checkpoint_rejects_unsafe_pickle_objects(self):
        import hashlib

        class UnsafePayload:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unsafe.pt"
            torch.save({
                "format_version": 4,
                "step": 0,
                "model": {},
                "optimizer": {},
                "scaler": {},
                "model_config": {},
                "data_sha256": "0" * 64,
                "tokenizer_sha256": "0" * 64,
                "rng_state": {},
                "extra": UnsafePayload(),
            }, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            Path(str(checkpoint) + ".sha256").write_text(digest + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "safely deserialized"):
                load_checkpoint(checkpoint, map_location="cpu")

    def test_batch_shapes_and_shift(self):
        x, y = get_batch(torch.arange(100), 4, 8, torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
    def test_rejects_too_short_data(self):
        with self.assertRaises(ValueError):
            get_batch(torch.arange(9), 2, 8, torch.device("cpu"))
    def test_training_blocks_when_dataset_manifest_is_false(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            raw_dir = work / "raw"
            raw_dir.mkdir()
            (raw_dir / "fixture.txt").write_text("Synthetic test-only source text. " * 100, encoding="utf-8")
            dataset_manifest = work / "dataset_manifest.json"
            dataset_manifest.write_text(json.dumps({
                "project": "AETHRION-LLM-test",
                "training_allowed": False
            }), encoding="utf-8")
            source_manifest = work / "sources.json"
            source_manifest.write_text(json.dumps({"sources": [{
                "source_id": "synthetic-test-fixture",
                "path": "fixture.txt",
                "license": "Synthetic test fixture",
                "permitted_uses": "machine-learning training for unit-test fixture only",
                "allow_training": True
            }]}), encoding="utf-8")
            command = [
                sys.executable, "-m", "aethrion.train",
                "--config", str(root / "configs/small.json"),
                "--dataset-manifest", str(dataset_manifest),
                "--source-manifest", str(source_manifest),
                "--source-root", str(raw_dir),
                "--corpus-report", str(work / "not-created-report.json"),
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Training blocked", result.stdout + result.stderr)

    def test_training_checkpoint_and_resume_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data_path = work / "corpus.txt"
            tokenizer_path = work / "tokenizer.json"
            config_path = work / "config.json"
            output = work / "artifacts"
            raw_dir = work / "raw"
            raw_dir.mkdir()
            source_texts = []
            for source_index in range(3):
                source_text = "\n".join(
                    f"Synthetic AETHRION unit-test source {source_index} record number {index} with Turkish sample text."
                    for index in range(40)
                )
                filename = f"approved_fixture_{source_index}.txt"
                (raw_dir / filename).write_text(source_text, encoding="utf-8")
                source_texts.append({
                    "source_id": f"synthetic-test-fixture-{source_index}",
                    "path": filename,
                    "license": "Synthetic test fixture",
                    "permitted_uses": "machine-learning training for unit-test fixture only",
                    "allow_training": True,
                })
            dataset_manifest = work / "dataset_manifest.json"
            dataset_manifest.write_text(json.dumps({
                "project": "AETHRION-LLM-test",
                "training_allowed": True,
                "governance": {
                    "schema": "aethrion-human-review-v1",
                    "model_id": "base",
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
                }
            }), encoding="utf-8")
            source_manifest = work / "sources.json"
            source_manifest.write_text(json.dumps({"sources": source_texts}), encoding="utf-8")
            corpus_report = work / "corpus-report.json"
            build_corpus(
                input_dir=raw_dir,
                manifest_path=source_manifest,
                output_path=data_path,
                report_path=corpus_report,
                dataset_manifest_path=dataset_manifest,
            )
            train_path = data_path.with_name("corpus-train.txt")
            tokenizer_command = [
                sys.executable, "-m", "aethrion.tokenizer",
                "--train", str(train_path), "--corpus", str(data_path),
                "--output", str(tokenizer_path), "--vocab-size", "256",
                "--dataset-manifest", str(dataset_manifest),
                "--source-manifest", str(source_manifest),
                "--source-root", str(raw_dir),
                "--corpus-report", str(corpus_report),
            ]
            tokenized = subprocess.run(
                tokenizer_command, cwd=root, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(tokenized.returncode, 0, tokenized.stdout + tokenized.stderr)
            self.assertTrue(Path(str(tokenizer_path) + ".provenance.json").is_file())
            config = {
                "seed": 42, "data_path": str(data_path), "tokenizer_path": str(tokenizer_path),
                "output_dir": str(output), "vocab_size": 256, "block_size": 8,
                "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0,
                "batch_size": 1, "gradient_accumulation_steps": 1,
                "learning_rate": 0.001, "min_learning_rate": 0.0001,
                "weight_decay": 0.0, "max_steps": 2, "warmup_steps": 1,
                "eval_interval": 1, "eval_batches": 1, "save_interval": 1,
                "grad_clip": 1.0, "amp": False
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = [
                sys.executable, "-m", "aethrion.train",
                "--config", str(config_path),
                "--dataset-manifest", str(dataset_manifest),
                "--source-manifest", str(source_manifest),
                "--source-root", str(raw_dir),
                "--corpus-report", str(corpus_report),
            ]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            checkpoint_path = output / "checkpoint_last.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((output / "progress.json").is_file())
            self.assertTrue((output / "metrics.jsonl").is_file())
            config["max_steps"] = 3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            resumed = subprocess.run(command + ["--resume", str(checkpoint_path)],
                                     cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["format_version"], 4)
            self.assertEqual(checkpoint["training_config"]["schedule_max_steps"], 2)
            self.assertTrue(Path(str(checkpoint_path) + ".sha256").is_file())

            config["max_steps"] = 4
            config["batch_size"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            changed_settings = subprocess.run(
                command + ["--resume", str(checkpoint_path)],
                cwd=root, capture_output=True, text=True, timeout=120
            )
            self.assertNotEqual(changed_settings.returncode, 0)
            self.assertIn("training settings changed", changed_settings.stdout + changed_settings.stderr)

            config["batch_size"] = 1
            config["max_steps"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            shorter_horizon = subprocess.run(
                command + ["--resume", str(checkpoint_path)],
                cwd=root, capture_output=True, text=True, timeout=120
            )
            self.assertNotEqual(shorter_horizon.returncode, 0)
            self.assertIn("max_steps cannot be lower", shorter_horizon.stdout + shorter_horizon.stderr)

            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = load_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(recovered["step"], 1)
            self.assertTrue(any("recovering from" in str(item.message) for item in caught))

            data_path.write_text(data_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
            tampered = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("prepared corpus SHA-256", tampered.stdout + tampered.stderr)


    def test_checkpoint_recovers_previous_generation_when_rng_snapshot_is_malformed(self):
        import hashlib
        from aethrion.checkpoint import save_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            payload = {
                "format_version": 4, "step": 0, "model": {}, "optimizer": {},
                "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
                "tokenizer_sha256": "1" * 64, "rng_state": capture_rng_state(),
                "training_config": {"stage": "test-fixture"},
            }
            save_checkpoint(payload, checkpoint)
            save_checkpoint({**payload, "step": 1}, checkpoint)
            malformed = {**payload, "step": 2, "rng_state": {}}
            torch.save(malformed, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            Path(str(checkpoint) + ".sha256").write_text(digest + "\n", encoding="ascii")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = load_checkpoint(checkpoint, map_location="cpu")
            self.assertEqual(recovered["step"], 0)
            self.assertTrue(any("recovering from" in str(item.message) for item in caught))

    def test_checkpoint_recovers_previous_generation_when_latest_safe_load_fails(self):
        import hashlib
        from aethrion.checkpoint import save_checkpoint

        class UnsafePayload:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            base = {
                "format_version": 4,
                "step": 0,
                "model": {},
                "optimizer": {},
                "scaler": {},
                "model_config": {},
                "data_sha256": "0" * 64,
                "tokenizer_sha256": "1" * 64,
                "rng_state": capture_rng_state(),
                "training_config": {"stage": "test-fixture"},
            }
            save_checkpoint(base, checkpoint)
            save_checkpoint({**base, "step": 1}, checkpoint)

            # The newest file has a matching checksum, but contains a pickle
            # global rejected by weights_only=True. Recovery must stay safe.
            unsafe = {**base, "step": 2, "extra": UnsafePayload()}
            torch.save(unsafe, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            Path(str(checkpoint) + ".sha256").write_text(digest + "\n", encoding="ascii")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = load_checkpoint(checkpoint, map_location="cpu")

            self.assertEqual(recovered["step"], 0)
            self.assertTrue(any("safely loaded" in str(item.message) for item in caught))



    def test_first_checkpoint_save_split_write_fails_closed_without_previous_generation(self):
        from unittest.mock import patch
        import aethrion.checkpoint as checkpoint_module
        from aethrion.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            payload = {
                "format_version": 4, "step": 0, "model": {}, "optimizer": {},
                "scaler": {}, "model_config": {}, "data_sha256": "0" * 64,
                "tokenizer_sha256": "1" * 64, "rng_state": capture_rng_state(),
                "training_config": {"seed": 7},
            }
            sidecar = Path(str(checkpoint) + ".sha256")
            original_replace = checkpoint_module.os.replace

            def fail_first_live_sidecar(source, destination):
                if Path(destination) == sidecar:
                    raise OSError("simulated first-generation sidecar interruption")
                return original_replace(source, destination)

            with patch("aethrion.checkpoint.os.replace", side_effect=fail_first_live_sidecar):
                with self.assertRaisesRegex(OSError, "first-generation sidecar interruption"):
                    save_checkpoint(payload, checkpoint)

            self.assertTrue(checkpoint.is_file())
            self.assertFalse(sidecar.exists())
            self.assertFalse(Path(str(checkpoint) + ".previous").exists())
            with self.assertRaisesRegex(ValueError, "integrity file is missing"):
                load_checkpoint(checkpoint, map_location="cpu")

    def test_checkpoint_loader_never_promotes_orphaned_temp_files_after_crash(self):
        from aethrion.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            temporary = Path(str(checkpoint) + ".tmp")
            sidecar_temporary = Path(str(checkpoint) + ".sha256.tmp")
            temporary.write_bytes(b"partial checkpoint from terminated writer")
            sidecar_temporary.write_text("0" * 64 + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
                load_checkpoint(checkpoint, map_location="cpu")

            self.assertTrue(temporary.is_file())
            self.assertTrue(sidecar_temporary.is_file())
            self.assertFalse(checkpoint.exists())
            self.assertFalse(Path(str(checkpoint) + ".previous").exists())


if __name__ == "__main__":
    unittest.main()
