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
            text = "AETHRION"
            self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
