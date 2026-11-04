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

if __name__ == "__main__":
    unittest.main()
