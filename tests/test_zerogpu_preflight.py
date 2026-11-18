import json
import unittest
from pathlib import Path

import torch

from aethrion.zerogpu_preflight import (
    MODEL_IDS, MODEL_MANIFEST, SPECIALIST_CONFIGS,
    _build_preflight_model, _synthetic_batch,
)


class ZeroGPUPreflightTests(unittest.TestCase):
    def test_preflight_covers_chat_and_all_ten_specialists(self):
        self.assertEqual(len(MODEL_IDS), 11)
        self.assertEqual(MODEL_IDS[0], "chat")
        self.assertEqual(set(MODEL_IDS[1:]), set(SPECIALIST_CONFIGS))

    def test_preflight_uses_exact_canonical_manifest_configurations(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(MODEL_MANIFEST, manifest["models"])
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                self.assertEqual(
                    MODEL_MANIFEST[model_id]["model_config"],
                    manifest["models"][model_id]["model_config"],
                )

    def test_all_eleven_models_are_constructed_on_the_requested_device(self):
        device = torch.device("cpu")
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                model = _build_preflight_model(model_id, device)
                parameters = list(model.parameters())
                self.assertTrue(parameters)
                self.assertTrue(all(parameter.device == device for parameter in parameters))
                del model

    def test_every_specialist_has_a_well_formed_disposable_batch(self):
        device = torch.device("cpu")
        for model_id in MODEL_IDS[1:]:
            with self.subTest(model=model_id):
                batch = _synthetic_batch(model_id, device)
                self.assertTrue(batch)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in batch.values()))
                self.assertTrue(all(value.device.type == "cpu" for value in batch.values()))

    def test_asr_preflight_targets_match_training_loss_batch_contract(self):
        batch = _synthetic_batch("asr", torch.device("cpu"))
        self.assertEqual(batch["targets"].ndim, 2)
        self.assertEqual(batch["targets"].shape[0], 1)
        self.assertEqual(batch["target_lengths"].shape, (1,))
        self.assertEqual(batch["input_lengths"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
