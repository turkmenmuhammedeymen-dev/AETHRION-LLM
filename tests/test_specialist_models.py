import unittest
import torch

from aethrion.code_model import AETHRIONCodeLM, CodeModelConfig
from aethrion.vision_model import AETHRIONVision, VisionConfig
from aethrion.image_model import AETHRIONImageGenerator, ImageModelConfig
from aethrion.asr_model import AETHRIONASR, ASRConfig
from aethrion.tts_model import AETHRIONTTS, AETHRIONVocoder, TTSConfig
from aethrion.model_3d import AETHRION3D, Model3DConfig, point_cloud_to_obj
from aethrion.research_model import AETHRIONResearch, ResearchConfig
from aethrion.document_model import AETHRIONDocument, DocumentConfig
from aethrion.safety_model import AETHRIONSafe, SafetyConfig


class SpecialistModelTests(unittest.TestCase):
    def test_code_model_has_independent_forward_and_generation(self):
        model = AETHRIONCodeLM(CodeModelConfig(
            vocab_size=300, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
        ))
        tokens = torch.randint(0, 300, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 300))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(model.generate(tokens[:, :3], 2).shape), (2, 5))

    def test_vision_model_contrastive_objective(self):
        model = AETHRIONVision(VisionConfig(
            image_size=32, patch_size=8, vocab_size=300, max_text_length=12,
            width=32, layers=1, heads=4, projection_dim=16, dropout=0.0
        ))
        images = torch.randn(2, 3, 32, 32)
        ids = torch.randint(0, 300, (2, 7))
        output = model(images, ids)
        self.assertEqual(tuple(output["image_features"].shape), (2, 16))
        self.assertEqual(tuple(output["logits_per_image"].shape), (2, 2))
        self.assertTrue(torch.isfinite(output["loss"]).item())

    def test_image_diffusion_objective(self):
        model = AETHRIONImageGenerator(ImageModelConfig(
            vocab_size=300, image_size=16, text_length=12, width=16,
            time_dim=32, diffusion_steps=20
        ))
        images = torch.randn(2, 3, 16, 16)
        ids = torch.randint(0, 300, (2, 6))
        timestep = torch.tensor([1, 7])
        noise = torch.randn_like(images)
        predicted = model(images, ids, timestep)
        self.assertEqual(tuple(predicted.shape), tuple(images.shape))
        loss = model.diffusion_loss(images, ids, timestep, noise)
        self.assertTrue(torch.isfinite(loss).item())

    def test_asr_ctc_objective_and_lengths(self):
        model = AETHRIONASR(ASRConfig(
            vocab_size=300, n_mels=16, width=32, layers=1, heads=4, dropout=0.0
        ))
        features = torch.randn(2, 40, 16)
        input_lengths = torch.tensor([40, 32])
        log_probs, encoded_lengths = model(features, input_lengths)
        self.assertEqual(tuple(log_probs.shape), (10, 2, 300))
        self.assertEqual(encoded_lengths.tolist(), [10, 8])
        targets = torch.tensor([1, 2, 3, 4, 5])
        loss = model.ctc_loss(features, targets, input_lengths, torch.tensor([3, 2]))
        self.assertTrue(torch.isfinite(loss).item())

    def test_tts_acoustic_model_and_vocoder(self):
        model = AETHRIONTTS(TTSConfig(
            vocab_size=300, mel_bins=16, width=32, layers=1, heads=4, dropout=0.0
        ))
        ids = torch.randint(0, 300, (2, 5))
        alignment = torch.randint(0, 5, (2, 12))
        mels = torch.randn(2, 12, 16)
        prediction = model(ids, alignment)
        self.assertEqual(tuple(prediction.shape), tuple(mels.shape))
        self.assertTrue(torch.isfinite(model.acoustic_loss(ids, alignment, mels)).item())
        vocoder = AETHRIONVocoder(mel_bins=16, channels=32, hop_length=32)
        waveform = vocoder(mels.transpose(1, 2))
        self.assertEqual(tuple(waveform.shape), (2, 12 * 32))

    def test_3d_point_cloud_diffusion_and_obj_export(self):
        model = AETHRION3D(Model3DConfig(
            vocab_size=300, width=32, points=32, layers=1, heads=4, diffusion_steps=20
        ))
        points = torch.randn(2, 32, 3)
        ids = torch.randint(0, 300, (2, 6))
        timesteps = torch.tensor([2, 9])
        output = model(points, ids, timesteps)
        self.assertEqual(tuple(output.shape), tuple(points.shape))
        self.assertTrue(torch.isfinite(model.diffusion_loss(points, ids, timesteps)).item())
        obj = point_cloud_to_obj(points[0])
        self.assertTrue(obj.startswith("# AETHRION point cloud"))
        self.assertEqual(sum(line.startswith("v ") for line in obj.splitlines()), 32)

    def test_research_model_dual_encoder_loss_and_retrieval(self):
        model = AETHRIONResearch(ResearchConfig(
            vocab_size=300, max_length=16, width=32, layers=1, heads=4, embedding_dim=16
        ))
        q = torch.randint(0, 300, (3, 8))
        p = torch.randint(0, 300, (3, 10))
        output = model(q, p)
        self.assertTrue(torch.isfinite(output["loss"]).item())
        scores, indexes = model.retrieve(q[:1], p, top_k=2)
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertEqual(tuple(indexes.shape), (1, 2))

    def test_document_layout_model_multitask_loss(self):
        model = AETHRIONDocument(DocumentConfig(
            vocab_size=300, layout_vocab_size=8, max_length=16,
            width=32, layers=1, heads=4, tag_classes=6
        ))
        ids = torch.randint(0, 300, (2, 10))
        boxes = torch.rand(2, 10, 4)
        layouts = torch.randint(0, 8, (2, 10))
        output = model(ids, boxes, layouts)
        self.assertEqual(tuple(output["tag_logits"].shape), (2, 10, 6))
        tags = torch.randint(0, 6, (2, 10))
        loss = model.multitask_loss(output, tag_labels=tags,
                                    start_positions=torch.tensor([1, 2]),
                                    end_positions=torch.tensor([4, 5]))
        self.assertTrue(torch.isfinite(loss).item())

    def test_safety_model_returns_three_decision_candidates(self):
        model = AETHRIONSafe(SafetyConfig(
            vocab_size=300, max_length=16, width=32, layers=1, heads=4,
            reason_classes=5
        ))
        ids = torch.randint(0, 300, (2, 8))
        output = model(ids)
        self.assertEqual(tuple(output["decision_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["reason_logits"].shape), (2, 5))
        prediction = model.predict_candidate(ids)
        self.assertEqual(len(prediction["decision"]), 2)
        self.assertTrue(all(item in {"ALLOW", "REVIEW", "BLOCK"} for item in prediction["decision"]))


if __name__ == "__main__":
    unittest.main()
