"""CPU forward/loss smoke tests for all native AETHRION model families.

These tests validate architecture wiring only. They use random synthetic tensors,
do not train or publish models, and do not alter production model status.
"""
import unittest

import torch

from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.specialist_train import _build_model, _loss


class NativeModelFamilySmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.batch = 2
        self.tokens = torch.randint(1, 200, (self.batch, 6), dtype=torch.long)

    def _run_loss(self, model_id, config, batch):
        model = _build_model(model_id, config)
        model.eval()
        loss = _loss(model, model_id, batch)
        self.assertEqual(loss.ndim, 0, model_id)
        self.assertTrue(torch.isfinite(loss).item(), model_id)
        loss.backward()
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()),
            model_id,
        )

    def test_base_chat_forward_and_loss(self):
        model = AETHRIONLM(ModelConfig(
            vocab_size=256, block_size=8, n_layer=1, n_head=2,
            n_embd=16, dropout=0.0,
        ))
        logits, loss = model(self.tokens, self.tokens)
        self.assertEqual(tuple(logits.shape), (self.batch, 6, 256))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_code_forward_and_loss(self):
        self._run_loss("code", {
            "vocab_size": 256, "block_size": 8, "n_layer": 1,
            "n_head": 2, "n_embd": 16, "dropout": 0.0,
        }, {"token_ids": self.tokens})

    def test_vision_forward_and_loss(self):
        self._run_loss("vision", {
            "image_size": 32, "patch_size": 8, "vocab_size": 256,
            "max_text_length": 8, "width": 16, "layers": 1,
            "heads": 2, "projection_dim": 8, "dropout": 0.0,
        }, {
            "images": torch.randn(self.batch, 3, 32, 32),
            "token_ids": self.tokens,
        })

    def test_image_forward_and_loss(self):
        self._run_loss("image", {
            "vocab_size": 256, "image_size": 16, "text_length": 8,
            "width": 16, "time_dim": 16, "diffusion_steps": 4,
        }, {
            "images": torch.randn(self.batch, 3, 16, 16),
            "token_ids": self.tokens,
        })

    def test_asr_forward_and_loss(self):
        self._run_loss("asr", {
            "vocab_size": 32, "n_mels": 16, "width": 16,
            "layers": 1, "heads": 2, "dropout": 0.0, "blank_id": 0,
        }, {
            "features": torch.randn(self.batch, 16, 16),
            "input_lengths": torch.tensor([16, 16]),
            "targets": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "target_lengths": torch.tensor([2, 2]),
        })

    def test_tts_forward_and_loss(self):
        self._run_loss("tts", {
            "vocab_size": 256, "mel_bins": 16, "width": 16,
            "layers": 1, "heads": 2, "dropout": 0.0,
        }, {
            "token_ids": self.tokens,
            "frame_to_token": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            "target_mels": torch.randn(self.batch, 4, 16),
            "durations": torch.ones(self.batch, 6, dtype=torch.long),
        })

    def test_vocoder_forward_and_loss(self):
        self._run_loss("vocoder", {
            "mel_bins": 16, "channels": 16, "hop_length": 4,
        }, {
            # _loss transposes [batch, frames, mel_bins] to the vocoder layout.
            "target_mels": torch.randn(self.batch, 4, 16),
            "target_waveforms": torch.randn(self.batch, 16),
        })

    def test_3d_forward_and_loss(self):
        self._run_loss("3d", {
            "vocab_size": 256, "width": 16, "points": 16,
            "layers": 1, "heads": 2, "diffusion_steps": 4,
        }, {
            "points": torch.randn(self.batch, 16, 3),
            "token_ids": self.tokens,
        })

    def test_research_forward_and_loss(self):
        self._run_loss("research", {
            "vocab_size": 256, "max_length": 8, "width": 16,
            "layers": 1, "heads": 2, "embedding_dim": 8,
        }, {
            "query_ids": self.tokens,
            "passage_ids": self.tokens.flip(dims=[1]),
        })

    def test_document_forward_and_loss(self):
        self._run_loss("document", {
            "vocab_size": 256, "layout_vocab_size": 8, "max_length": 8,
            "width": 16, "layers": 1, "heads": 2, "tag_classes": 4,
        }, {
            "token_ids": self.tokens,
            "boxes": torch.rand(self.batch, 6, 4),
            "layout_ids": torch.zeros(self.batch, 6, dtype=torch.long),
            "tag_labels": torch.zeros(self.batch, 6, dtype=torch.long),
            "start_positions": torch.tensor([0, 1]),
            "end_positions": torch.tensor([2, 3]),
        })

    def test_safety_forward_and_loss(self):
        self._run_loss("safety", {
            "vocab_size": 256, "max_length": 8, "width": 16,
            "layers": 1, "heads": 2, "reason_classes": 4, "decisions": 3,
        }, {
            "token_ids": self.tokens,
            "decision_labels": torch.tensor([0, 1]),
            "reason_labels": torch.zeros(self.batch, 4),
        })


if __name__ == "__main__":
    unittest.main()
