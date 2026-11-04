import unittest
import torch
from aethrion.train import get_batch

class TrainingUtilityTests(unittest.TestCase):
    def test_batch_shapes_and_shift(self):
        x, y = get_batch(torch.arange(100), 4, 8, torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
    def test_rejects_too_short_data(self):
        with self.assertRaises(ValueError):
            get_batch(torch.arange(9), 2, 8, torch.device("cpu"))

if __name__ == "__main__":
    unittest.main()
