import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from aethrion.tokenizer import ByteBPETokenizer

class TokenizerTests(unittest.TestCase):
    def test_round_trip_utf8(self):
        text = "Merhaba dünya! Türkçe karakterler: ğüşiöç İslam."
        tokenizer = ByteBPETokenizer()
        tokenizer.train((text + "\n") * 5, vocab_size=300, min_frequency=2)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
    def test_save_load(self):
        tokenizer = ByteBPETokenizer()
        tokenizer.train(("aethrion model test\n") * 4, vocab_size=280)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            self.assertEqual(loaded.decode(loaded.encode("AETHRION")), "AETHRION")
    def test_rejects_negative_token_id(self):
        with self.assertRaises(ValueError):
            ByteBPETokenizer().decode([-1])
    def test_rejects_invalid_merge_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"format":"aethrion-byte-bpe-v1","merges":[[999,1]]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ByteBPETokenizer.load(path)

    def test_tokenizer_cli_blocks_without_dataset_approval(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            corpus = work / "corpus.txt"
            corpus.write_text(("Synthetic test fixture only. " * 20), encoding="utf-8")
            raw = work / "raw"
            raw.mkdir()
            dataset = work / "dataset.json"
            dataset.write_text(json.dumps({"training_allowed": False}), encoding="utf-8")
            sources = work / "sources.json"
            sources.write_text(json.dumps({"sources": []}), encoding="utf-8")
            output = work / "tokenizer.json"
            command = [
                sys.executable, "-m", "aethrion.tokenizer",
                "--train", str(corpus), "--corpus", str(corpus), "--output", str(output),
                "--dataset-manifest", str(dataset), "--source-manifest", str(sources),
                "--source-root", str(raw), "--corpus-report", str(work / "missing-report.json"),
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Training blocked", result.stdout + result.stderr)
            self.assertFalse(output.exists(), "Blocked tokenizer command must not write an artifact.")

if __name__ == "__main__":
    unittest.main()
