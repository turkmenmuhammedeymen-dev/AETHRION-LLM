import unittest

import torch

from aethrion.model import AETHRIONLM, ModelConfig


class ModelTests(unittest.TestCase):
    def test_forward_and_loss(self):
        config = ModelConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2,
                             n_embd=32, dropout=0.0)
        model = AETHRIONLM(config)
        tokens = torch.randint(0, 300, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 300))
        self.assertTrue(torch.isfinite(loss).item())

    def test_rejects_long_context(self):
        model = AETHRIONLM(ModelConfig(vocab_size=300, block_size=8, n_layer=1,
                                       n_head=1, n_embd=16))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 9), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
