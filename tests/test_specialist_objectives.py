"""Synthetic objective smoke tests for every independently trained specialist."""
import unittest

import numpy as np
import torch

from aethrion.specialist_train import _build_model, _loss, _to_batch


class SpecialistObjectiveSmokeTests(unittest.TestCase):
    def _case(self, model_id):
        rng = np.random.default_rng(123)
        n = 2
        configs = {
            "code": {"vocab_size": 256, "block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "vision": {"image_size": 32, "patch_size": 8, "vocab_size": 256, "max_text_length": 4, "width": 16, "layers": 1, "heads": 2, "projection_dim": 8, "dropout": 0.0},
            "image": {"vocab_size": 256, "image_size": 16, "text_length": 4, "width": 16, "time_dim": 16, "diffusion_steps": 4},
            "asr": {"vocab_size": 256, "n_mels": 16, "width": 16, "layers": 1, "heads": 2, "dropout": 0.0, "blank_id": 0},
            "tts": {"vocab_size": 256, "mel_bins": 16, "width": 16, "layers": 1, "heads": 2, "dropout": 0.0},
            "vocoder": {"mel_bins": 16, "channels": 16, "hop_length": 4},
            "3d": {"vocab_size": 256, "width": 16, "points": 16, "layers": 1, "heads": 2, "diffusion_steps": 4},
            "research": {"vocab_size": 256, "max_length": 8, "width": 16, "layers": 1, "heads": 2, "embedding_dim": 8},
            "document": {"vocab_size": 256, "layout_vocab_size": 8, "max_length": 8, "width": 16, "layers": 1, "heads": 2, "tag_classes": 8},
            "safety": {"vocab_size": 256, "max_length": 8, "width": 16, "layers": 1, "heads": 2, "reason_classes": 16, "decisions": 3},
        }
        ids = rng.integers(0, 256, size=(n, 4), dtype=np.int64)
        arrays = {}
        if model_id == "code":
            arrays = {"token_ids": ids}
        elif model_id == "vision":
            arrays = {"images": rng.uniform(-1, 1, (n, 3, 32, 32)).astype(np.float32), "token_ids": ids}
        elif model_id == "image":
            arrays = {"images": rng.uniform(-1, 1, (n, 3, 16, 16)).astype(np.float32), "token_ids": ids}
        elif model_id == "asr":
            arrays = {
                "features": rng.normal(size=(n, 16, 16)).astype(np.float32),
                "input_lengths": np.asarray([16, 16], dtype=np.int64),
                "targets": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                "target_lengths": np.asarray([2, 2], dtype=np.int64),
            }
        elif model_id == "tts":
            arrays = {
                "token_ids": ids,
                "frame_to_token": np.tile(np.arange(4, dtype=np.int64), (n, 1)),
                "target_mels": rng.normal(size=(n, 4, 16)).astype(np.float32),
                "durations": np.ones((n, 4), dtype=np.int64),
            }
        elif model_id == "vocoder":
            arrays = {
                "target_mels": rng.normal(size=(n, 4, 16)).astype(np.float32),
                "target_waveforms": rng.uniform(-1, 1, (n, 16)).astype(np.float32),
            }
        elif model_id == "3d":
            arrays = {"points": rng.uniform(-1, 1, (n, 16, 3)).astype(np.float32), "token_ids": ids}
        elif model_id == "research":
            arrays = {"query_ids": ids, "passage_ids": ids.copy()}
        elif model_id == "document":
            arrays = {
                "token_ids": ids,
                "boxes": rng.uniform(0, 1, (n, 4, 4)).astype(np.float32),
                "layout_ids": rng.integers(0, 8, (n, 4), dtype=np.int64),
                "tag_labels": rng.integers(0, 8, (n, 4), dtype=np.int64),
            }
        elif model_id == "safety":
            reasons = np.zeros((n, 16), dtype=np.int64)
            reasons[:, 0] = 1
            arrays = {
                "token_ids": ids,
                "decision_labels": np.asarray([0, 2], dtype=np.int64),
                "reason_labels": reasons,
            }
        batch = _to_batch(arrays, sorted(arrays), torch.arange(n), torch.device("cpu"))
        return configs[model_id], batch

    def test_every_specialist_has_a_finite_forward_and_backward_objective(self):
        model_ids = ("code", "vision", "image", "asr", "tts", "vocoder", "3d", "research", "document", "safety")
        for model_id in model_ids:
            with self.subTest(model_id=model_id):
                config, batch = self._case(model_id)
                model = _build_model(model_id, config)
                loss = _loss(model, model_id, batch)
                self.assertTrue(torch.isfinite(loss).item(), f"{model_id} loss is not finite")
                loss.backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters()),
                    f"{model_id} did not produce any parameter gradients",
                )


if __name__ == "__main__":
    unittest.main()
