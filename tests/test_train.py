import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import torch
from aethrion.train import get_batch
from aethrion.tokenizer import ByteBPETokenizer
from aethrion.prepare_data import build_corpus
from aethrion.checkpoint import load_checkpoint

class TrainingUtilityTests(unittest.TestCase):
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
            source_text = "\n".join(
                f"Synthetic AETHRION unit-test record number {index} with Turkish sample text."
                for index in range(120)
            )
            (raw_dir / "approved_fixture.txt").write_text(source_text, encoding="utf-8")
            dataset_manifest = work / "dataset_manifest.json"
            dataset_manifest.write_text(json.dumps({
                "project": "AETHRION-LLM-test",
                "training_allowed": True
            }), encoding="utf-8")
            source_manifest = work / "sources.json"
            source_manifest.write_text(json.dumps({"sources": [{
                "source_id": "synthetic-test-fixture",
                "path": "approved_fixture.txt",
                "license": "Synthetic test fixture",
                "permitted_uses": "unit tests only",
                "allow_training": True
            }]}), encoding="utf-8")
            corpus_report = work / "corpus-report.json"
            build_corpus(
                input_dir=raw_dir,
                manifest_path=source_manifest,
                output_path=data_path,
                report_path=corpus_report,
                dataset_manifest_path=dataset_manifest,
            )
            tokenizer = ByteBPETokenizer()
            tokenizer.train(data_path.read_text(encoding="utf-8"), vocab_size=256)
            tokenizer.save(tokenizer_path)
            config = {
                "seed": 42, "data_path": str(data_path), "tokenizer_path": str(tokenizer_path),
                "output_dir": str(output), "vocab_size": 256, "block_size": 8,
                "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0,
                "batch_size": 1, "gradient_accumulation_steps": 1,
                "learning_rate": 0.001, "min_learning_rate": 0.0001,
                "weight_decay": 0.0, "max_steps": 2, "warmup_steps": 1,
                "eval_interval": 1, "eval_batches": 1, "save_interval": 1,
                "train_split": 0.8, "grad_clip": 1.0, "amp": False
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
            config["max_steps"] = 3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            resumed = subprocess.run(command + ["--resume", str(checkpoint_path)],
                                     cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["format_version"], 3)
            self.assertTrue(Path(str(checkpoint_path) + ".sha256").is_file())

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

if __name__ == "__main__":
    unittest.main()
