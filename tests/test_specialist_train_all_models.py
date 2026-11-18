"""One-step, from-random-init training smoke test for every specialist head.

All samples are generated numeric fixtures in temporary directories. They are
not production data and never enter the repository or a released checkpoint.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from aethrion.specialist_train import train_specialist

MODEL_CONFIGS = {
    "code": {"vocab_size":256,"block_size":8,"n_layer":1,"n_head":4,"n_embd":16,"dropout":0.0},
    "vision": {"image_size":32,"patch_size":8,"vocab_size":256,"max_text_length":8,"width":16,"layers":1,"heads":4,"projection_dim":16,"dropout":0.0},
    "image": {"vocab_size":256,"image_size":32,"text_length":8,"width":16,"time_dim":16,"diffusion_steps":4},
    "asr": {"vocab_size":256,"n_mels":16,"width":16,"layers":1,"heads":4,"dropout":0.0,"blank_id":0},
    "tts": {"vocab_size":256,"mel_bins":16,"width":16,"layers":1,"heads":4,"dropout":0.0},
    "vocoder": {"mel_bins":16,"channels":16,"hop_length":4},
    "3d": {"vocab_size":256,"width":16,"points":16,"layers":1,"heads":4,"diffusion_steps":4},
    "research": {"vocab_size":256,"max_length":8,"width":16,"layers":1,"heads":4,"embedding_dim":16},
    "document": {"vocab_size":256,"layout_vocab_size":4,"max_length":8,"width":16,"layers":1,"heads":4,"tag_classes":4},
    "safety": {"vocab_size":256,"max_length":8,"width":16,"layers":1,"heads":4,"reason_classes":4,"decisions":3},
}
MODALITIES = {
    "code":"code","vision":"vision","image":"image","asr":"audio","tts":"audio",
    "vocoder":"audio","3d":"3d","research":"text","document":"document","safety":"text",
}


class SpecialistFamilyTrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _arrays(self, model_id):
        rng = np.random.default_rng(20261117)
        n = 225 if model_id == "safety" else 6
        splits = (
            ["train"] * 75 + ["validation"] * 75 + ["test"] * 75
            if model_id == "safety"
            else ["train", "train", "validation", "validation", "test", "test"]
        )
        arrays = {
            "sample_id": np.asarray([f"{model_id}-{i}" for i in range(n)]),
            "group_id": np.asarray([
                hashlib.sha256(f"{model_id}-group-{i}".encode()).hexdigest()
                for i in range(n)
            ]),
            "split": np.asarray(splits),
        }
        ids = rng.integers(1, 255, size=(n, 8), dtype=np.int64)
        if model_id == "code":
            arrays["token_ids"] = ids
        elif model_id in {"vision", "image"}:
            arrays["images"] = rng.integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)
            arrays["token_ids"] = ids
        elif model_id == "asr":
            arrays.update({
                "features": rng.normal(size=(n, 32, 16)).astype(np.float32),
                "input_lengths": np.full(n, 32, dtype=np.int64),
                "targets": rng.integers(1, 255, size=(n, 2), dtype=np.int64),
                "target_lengths": np.full(n, 2, dtype=np.int64),
            })
        elif model_id == "tts":
            arrays.update({
                "token_ids": ids[:, :4],
                "token_mask": np.ones((n, 4), dtype=np.bool_),
                "token_lengths": np.full(n, 4, dtype=np.int64),
                "frame_to_token": np.tile(np.arange(4, dtype=np.int64), (n, 1)),
                "frame_mask": np.ones((n, 4), dtype=np.bool_),
                "frame_lengths": np.full(n, 4, dtype=np.int64),
                "target_mels": rng.normal(size=(n, 4, 16)).astype(np.float32),
                "durations": np.ones((n, 4), dtype=np.int64),
            })
        elif model_id == "vocoder":
            arrays.update({
                "target_mels": rng.normal(size=(n, 4, 16)).astype(np.float32),
                "target_waveforms": rng.normal(size=(n, 16)).astype(np.float32),
                "frame_lengths": np.full(n, 4, dtype=np.int64),
                "waveform_mask": np.ones((n, 16), dtype=np.bool_),
            })
        elif model_id == "3d":
            arrays.update({
                "points": rng.normal(size=(n, 16, 3)).astype(np.float32),
                "token_ids": ids[:, :4],
            })
        elif model_id == "research":
            arrays.update({
                "query_ids": ids[:, :4],
                "passage_ids": ids[:, :6],
            })
        elif model_id == "document":
            arrays.update({
                "token_ids": ids[:, :4],
                "boxes": rng.random(size=(n, 4, 4)).astype(np.float32),
                "layout_ids": rng.integers(0, 4, size=(n, 4), dtype=np.int64),
                "tag_labels": rng.integers(0, 4, size=(n, 4), dtype=np.int64),
            })
        elif model_id == "safety":
            arrays.update({
                "token_ids": ids[:, :4],
                "decision_labels": np.tile(np.asarray([0, 1, 2] * 25, dtype=np.int64), 3),
                "reason_labels": rng.integers(0, 2, size=(n, 4), dtype=np.int64),
            })
        return arrays

    def test_model_data_config_compatibility_is_checked_for_all_specialists(self):
        from aethrion.specialist_train import _validate_model_data_compatibility

        mismatches = {
            "code": {"block_size": 1},
            "vision": {"image_size": 64},
            "image": {"image_size": 64},
            "asr": {"n_mels": 80},
            "tts": {"mel_bins": 80},
            "vocoder": {"hop_length": 8},
            "3d": {"points": 32},
            "research": {"max_length": 4},
            "document": {"max_length": 2},
            "safety": {"max_length": 2},
        }
        for model_id, model_config in MODEL_CONFIGS.items():
            with self.subTest(model_id=model_id):
                arrays = self._arrays(model_id)
                _validate_model_data_compatibility(model_id, arrays, model_config)
                if model_id == "vocoder":
                    arrays["waveform_mask"][0, 12:] = False
                    _validate_model_data_compatibility(model_id, arrays, model_config)
                invalid = dict(model_config)
                invalid.update(mismatches[model_id])
                with self.assertRaises(ValueError):
                    _validate_model_data_compatibility(model_id, arrays, invalid)

    def test_each_specialist_completes_one_temporary_smoke_step_from_random_init(self):
        for model_id in MODEL_CONFIGS:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "synthetic_source.txt"
                source.write_text(
                    "Generated numeric fixture only; no human, private or third-party records.",
                    encoding="utf-8",
                )
                dataset = root / "synthetic_data.npz"
                np.savez(dataset, **self._arrays(model_id))
                tokenizer_path = root / "tokenizer.json"
                if model_id != "vocoder":
                    from aethrion.tokenizer import ByteBPETokenizer
                    ByteBPETokenizer().save(tokenizer_path)
                audio = model_id in {"asr","tts","vocoder"}
                source_record = {
                    "source_id":"generated-test-fixture",
                    "source_uri":"urn:aethrion:test:generated-fixture",
                    "source_path":source.name,
                    "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                    "license":"AETHRION synthetic test fixture",
                    "permitted_use":"Temporary unit-test training smoke test only",
                    "evidence_uri":"urn:aethrion:test:fixture-provenance",
                    "allow_training":True,
                    "privacy_reviewed":True,
                }
                if audio:
                    source_record["speaker_consent_verified"] = True
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({
                    "format":"aethrion-specialist-dataset-v1",
                    "dataset_id":f"temporary-synthetic-{model_id}",
                    "model_id":model_id,
                    "modality":MODALITIES[model_id],
                    "training_allowed":True,
                    "owner_approval":True,
                    "privacy_reviewed":True,
                    "speaker_consent_reviewed":audio,
                    "governance":{
                        "schema":"aethrion-human-review-v1",
                        "model_id":model_id,
                        "approval_status":"approved",
                        "author_role":"unit-test author",
                        "authorship_attested":True,
                        "first_party_origin_attested":True,
                        "rights_holder_role":"unit-test rights holder",
                        "rights_attested":True,
                        "source_rights_reviewed":True,
                        "privacy_reviewed":True,
                        "quality_reviewed":True,
                        "reviewer_role":"unit-test reviewer",
                        "reviewed_at":"2026-01-01T00:00:00Z",
                        "evidence_uri":"unit-test://human-review",
                        "speaker_consent_reviewed":audio,
                        "safety_reviewed":model_id == "safety",
                    },
                    "safety_min_per_class":25 if model_id == "safety" else None,
                    "dataset_path":dataset.name,
                    "dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    **({"tokenizer_path": tokenizer_path.name,
                        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()}
                       if model_id != "vocoder" else {}),
                    "sources":[source_record],
                    "synthetic_test_only":True,
                }), encoding="utf-8")
                config = {
                    "model_id":model_id,
                    "model_config":MODEL_CONFIGS[model_id],
                    "output_dir":str(root / "artifacts" / model_id),
                    "seed":17,
                    "max_steps":1,
                    "batch_size":2,
                    "learning_rate":0.001,
                    "weight_decay":0.0,
                    "grad_clip":1.0,
                    "eval_interval":1,
                    "save_interval":1,
                }
                report = train_specialist(config, manifest)
                output = Path(config["output_dir"])
                checkpoint = output / "checkpoint_last.pt"
                self.assertEqual(report["status"], "training_complete_not_released")
                self.assertEqual(report["release_status"], "NOT_RELEASED")
                self.assertTrue(checkpoint.is_file())
                self.assertTrue(Path(str(checkpoint) + ".sha256").is_file())
                self.assertTrue((output / "metrics.jsonl").is_file())
                self.assertTrue((output / "training_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
