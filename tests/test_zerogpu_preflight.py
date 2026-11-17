import unittest

import torch

from aethrion.zerogpu_preflight import MODEL_IDS, SPECIALIST_CONFIGS, _synthetic_batch


class ZeroGPUPreflightTests(unittest.TestCase):
    def test_preflight_covers_chat_and_all_ten_specialists(self):
        self.assertEqual(len(MODEL_IDS), 11)
        self.assertEqual(MODEL_IDS[0], "chat")
        self.assertEqual(set(MODEL_IDS[1:]), set(SPECIALIST_CONFIGS))

    def test_every_specialist_has_a_well_formed_disposable_batch(self):
        device = torch.device("cpu")
        for model_id in MODEL_IDS[1:]:
            with self.subTest(model=model_id):
                batch = _synthetic_batch(model_id, device)
                self.assertTrue(batch)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in batch.values()))
                self.assertTrue(all(value.device.type == "cpu" for value in batch.values()))


if __name__ == "__main__":
    unittest.main()
