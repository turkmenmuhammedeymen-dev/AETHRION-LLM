import unittest
import torch
from aethrion.model import AETHRIONLM, ModelConfig

class ModelTests(unittest.TestCase):
    def make_model(self):
        return AETHRIONLM(ModelConfig(vocab_size=300, block_size=16, n_layer=2,
                                      n_head=2, n_embd=32, dropout=0.0))
    def test_forward_and_loss(self):
        model = self.make_model()
        tokens = torch.randint(0, 300, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 300))
        self.assertTrue(torch.isfinite(loss).item())
    def test_rejects_long_context(self):
        with self.assertRaises(ValueError):
            self.make_model()(torch.zeros((1, 17), dtype=torch.long))
    def test_rejects_invalid_token_ids(self):
        with self.assertRaises(ValueError):
            self.make_model()(torch.tensor([[300]], dtype=torch.long))
    def test_generation_supports_batches_and_restores_mode(self):
        model = self.make_model()
        model.train()
        prompt = torch.randint(0, 300, (2, 5))
        result = model.generate(prompt, 3, top_k=10, top_p=0.9)
        self.assertEqual(tuple(result.shape), (2, 8))
        self.assertTrue(model.training)
    def test_generation_validates_sampling_parameters(self):
        model = self.make_model()
        prompt = torch.ones((1, 2), dtype=torch.long)
        with self.assertRaises(ValueError):
            model.generate(prompt, 2, top_k=0)
        with self.assertRaises(ValueError):
            model.generate(prompt, 2, top_p=0)

if __name__ == "__main__":
    unittest.main()
