import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from aethrion.train import get_batch
from aethrion.tokenizer import ByteBPETokenizer

class TrainingUtilityTests(unittest.TestCase):
    def test_batch_shapes_and_shift(self):
        x, y = get_batch(torch.arange(100), 4, 8, torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
    def test_rejects_too_short_data(self):
        with self.assertRaises(ValueError):
            get_batch(torch.arange(9), 2, 8, torch.device("cpu"))
    def test_training_checkpoint_and_resume_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data_path = work / "corpus.txt"
            tokenizer_path = work / "tokenizer.json"
            config_path = work / "config.json"
            output = work / "artifacts"
            data_path.write_text(("Türkçe AETHRION dil modeli eğitim denemesi. " * 120), encoding="utf-8")
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
            command = [sys.executable, "-m", "aethrion.train", "--config", str(config_path)]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            checkpoint_path = output / "checkpoint_last.pt"
            self.assertTrue(checkpoint_path.exists())
            config["max_steps"] = 3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            resumed = subprocess.run(command + ["--resume", str(checkpoint_path)],
                                     cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["format_version"], 2)

if __name__ == "__main__":
    unittest.main()
