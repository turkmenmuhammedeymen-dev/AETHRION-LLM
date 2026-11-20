"""Process-level SIGTERM recovery tests for the real BASE and specialist trainers.

All datasets and artifacts are temporary synthetic test fixtures. These tests do
not touch production data, model checkpoints, or the production model manifest.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

from aethrion.checkpoint import load_checkpoint
from aethrion.prepare_data import build_corpus
from aethrion.tokenizer import ByteBPETokenizer
from aethrion.specialist_train import _load_checkpoint


def _wait_for_file(path: Path, process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"trainer exited before readiness (code={process.returncode})\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.02)
    process.kill()
    stdout, stderr = process.communicate()
    raise AssertionError(f"trainer readiness timed out\nstdout={stdout}\nstderr={stderr}")


def _stop_process_and_collect(process: subprocess.Popen) -> tuple[str, str]:
    try:
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(f"trainer did not stop after SIGTERM\nstdout={stdout}\nstderr={stderr}")
    if process.returncode != 0:
        raise AssertionError(
            f"trainer failed after SIGTERM (code={process.returncode})\n"
            f"stdout={stdout}\nstderr={stderr}"
        )
    return stdout, stderr


class RealTrainingSigtermRecoveryTests(unittest.TestCase):
    def _base_fixture(self, root: Path) -> tuple[list[str], Path, Path]:
        raw = root / "raw"
        raw.mkdir()
        sources = []
        for source_index in range(3):
            filename = f"source_{source_index}.txt"
            text = "\n".join(
                f"Synthetic test-only AETHRION record {source_index}-{index} with Turkish sample text."
                for index in range(40)
            )
            (raw / filename).write_text(text, encoding="utf-8")
            sources.append({
                "source_id": f"sigterm-fixture-{source_index}",
                "path": filename,
                "license": "CC0-1.0",
                "permitted_uses": "unit-test fixture only; not production training data",
                "allow_training": True,
            })

        dataset_manifest = root / "dataset_manifest.json"
        dataset_manifest.write_text(json.dumps({
            "project": "AETHRION-LLM-process-test",
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
                "evidence_uri": "unit-test://synthetic-fixture",
            },
        }), encoding="utf-8")
        source_manifest = root / "sources.json"
        source_manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
        corpus = root / "corpus.txt"
        report = root / "corpus-report.json"
        build_corpus(
            input_dir=raw,
            manifest_path=source_manifest,
            output_path=corpus,
            report_path=report,
            dataset_manifest_path=dataset_manifest,
        )
        tokenizer = root / "tokenizer.json"
        tokenized = subprocess.run([
            sys.executable, "-m", "aethrion.tokenizer",
            "--train", str(corpus.with_name("corpus-train.txt")),
            "--corpus", str(corpus), "--output", str(tokenizer),
            "--vocab-size", "256",
            "--dataset-manifest", str(dataset_manifest),
            "--source-manifest", str(source_manifest),
            "--source-root", str(raw),
            "--corpus-report", str(report),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True,
           text=True, timeout=120)
        self.assertEqual(tokenized.returncode, 0, tokenized.stdout + tokenized.stderr)

        output = root / "base-artifacts"
        config = {
            "seed": 42, "data_path": str(corpus), "tokenizer_path": str(tokenizer),
            "output_dir": str(output), "vocab_size": 256, "block_size": 8,
            "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0,
            "batch_size": 1, "gradient_accumulation_steps": 1,
            "learning_rate": 0.001, "min_learning_rate": 0.0001,
            "weight_decay": 0.0, "max_steps": 10000, "warmup_steps": 1,
            "eval_interval": 1, "eval_batches": 1, "save_interval": 10000,
            "grad_clip": 1.0, "amp": False,
        }
        config_path = root / "base-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        command = [
            sys.executable, "-m", "aethrion.train", "--config", str(config_path),
            "--dataset-manifest", str(dataset_manifest),
            "--source-manifest", str(source_manifest),
            "--source-root", str(raw), "--corpus-report", str(report),
        ]
        return command, output, config_path

    def _specialist_fixture(self, root: Path) -> tuple[list[str], Path]:
        source = root / "specialist-source.txt"
        source.write_text("Synthetic test-only code fixture.", encoding="utf-8")
        dataset = root / "specialist-dataset.npz"
        np.savez(
            dataset,
            sample_id=np.asarray([f"s{i}" for i in range(6)]),
            group_id=np.asarray([
                hashlib.sha256(f"g{i}".encode()).hexdigest() for i in range(6)
            ]),
            split=np.asarray(["train", "train", "validation", "validation", "test", "test"]),
            token_ids=np.random.default_rng(42).integers(0, 256, size=(6, 12), dtype=np.int64),
        )
        tokenizer = root / "specialist-tokenizer.json"
        ByteBPETokenizer().save(tokenizer)
        manifest = root / "specialist-manifest.json"
        manifest.write_text(json.dumps({
            "format": "aethrion-specialist-dataset-v1",
            "dataset_id": "synthetic-sigterm-unit-test-only",
            "model_id": "code",
            "modality": "code",
            "training_allowed": True,
            "owner_approval": True,
            "privacy_reviewed": True,
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
                "evidence_uri": "unit-test://synthetic-review",
            },
            "dataset_path": dataset.name,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "tokenizer_path": tokenizer.name,
            "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "synthetic-fixture",
                "source_uri": "unit-test://synthetic",
                "source_path": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "CC0-1.0",
                "permitted_use": "Unit-test training fixture only; not production training data",
                "evidence_uri": "https://example.invalid/fixture-license",
                "allow_training": True,
                "privacy_reviewed": True,
            }],
        }), encoding="utf-8")
        output = root / "specialist-artifacts"
        config = {
            "model_id": "code",
            "model_config": {
                "vocab_size": 256, "block_size": 16, "n_layer": 1,
                "n_head": 2, "n_embd": 16, "dropout": 0.0,
            },
            "output_dir": str(output), "seed": 11, "max_steps": 10000,
            "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0,
            "grad_clip": 1.0, "eval_interval": 1, "save_interval": 10000,
        }
        config_path = root / "specialist-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        command = [
            sys.executable, "-m", "aethrion.specialist_train",
            "--config", str(config_path), "--manifest", str(manifest),
        ]
        return command, output

    def test_real_base_trainer_sigterm_checkpoint_and_resume(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            command, output, _ = self._base_fixture(Path(folder))
            process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            try:
                progress = output / "progress.json"
                _wait_for_file(progress, process)
                _stop_process_and_collect(process)
                progress_data = json.loads(progress.read_text(encoding="utf-8"))
                self.assertEqual(progress_data["status"], "interrupted")
                checkpoint = output / "checkpoint_last.pt"
                state = load_checkpoint(checkpoint, map_location="cpu")
                stopped_step = state["step"]
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())

                resumed = subprocess.run(
                    command + ["--resume", str(checkpoint), "--session-steps", "1"],
                    cwd=repo_root, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                resumed_state = load_checkpoint(checkpoint, map_location="cpu")
                self.assertGreater(resumed_state["step"], stopped_step)
                self.assertEqual(
                    json.loads(progress.read_text(encoding="utf-8"))["status"], "interrupted"
                )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_real_specialist_trainer_sigterm_checkpoint_and_resume(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            command, output = self._specialist_fixture(Path(folder))
            process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            try:
                metrics = output / "metrics.jsonl"
                _wait_for_file(metrics, process)
                _stop_process_and_collect(process)
                checkpoint = output / "checkpoint_last.pt"
                state = _load_checkpoint(checkpoint, torch.device("cpu"))
                stopped_step = state["step"]
                report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "training_interrupted_not_released")
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())

                resumed = subprocess.run(
                    command + ["--resume", str(checkpoint), "--session-steps", "1"],
                    cwd=repo_root, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                resumed_state = _load_checkpoint(checkpoint, torch.device("cpu"))
                self.assertGreater(resumed_state["step"], stopped_step)
                resumed_report = json.loads(
                    (output / "training_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    resumed_report["status"], "training_interrupted_not_released"
                )
                self.assertEqual(resumed_report["release_status"], "NOT_RELEASED")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


    def test_real_base_trainer_recovers_after_ungraceful_process_kill(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            command, output, _ = self._base_fixture(Path(folder))
            process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            try:
                checkpoint = output / "checkpoint_last.pt"
                checksum = Path(str(checkpoint) + ".sha256")
                _wait_for_file(checksum, process)
                os.kill(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=30)
                self.assertNotEqual(process.returncode, 0, stdout + stderr)

                crashed_state = load_checkpoint(checkpoint, map_location="cpu")
                crashed_step = crashed_state["step"]
                resumed = subprocess.run(
                    command + ["--resume", str(checkpoint), "--session-steps", "1"],
                    cwd=repo_root, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                recovered_state = load_checkpoint(checkpoint, map_location="cpu")
                self.assertGreater(recovered_state["step"], crashed_step)
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_real_specialist_trainer_recovers_after_ungraceful_process_kill(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            command, output = self._specialist_fixture(Path(folder))
            process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            try:
                checkpoint = output / "checkpoint_last.pt"
                checksum = Path(str(checkpoint) + ".sha256")
                _wait_for_file(checksum, process)
                os.kill(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=30)
                self.assertNotEqual(process.returncode, 0, stdout + stderr)

                crashed_state = _load_checkpoint(checkpoint, torch.device("cpu"))
                crashed_step = crashed_state["step"]
                resumed = subprocess.run(
                    command + ["--resume", str(checkpoint), "--session-steps", "1"],
                    cwd=repo_root, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                recovered_state = _load_checkpoint(checkpoint, torch.device("cpu"))
                self.assertGreater(recovered_state["step"], crashed_step)
                report = json.loads(
                    (output / "training_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(report["status"], "training_interrupted_not_released")
                self.assertEqual(report["release_status"], "NOT_RELEASED")
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
