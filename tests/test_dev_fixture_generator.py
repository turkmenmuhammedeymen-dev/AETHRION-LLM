import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from aethrion.specialist_prepare import prepare_text_dataset
from aethrion.specialist_prepare_visual import prepare_visual_dataset

MODEL_IDS = ("chat","code","vision","image","asr","tts","3d","research","document","safety","vocoder")


class DevFixtureGeneratorTests(unittest.TestCase):
    def test_generates_only_unapproved_smoke_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "scripts/generate_dev_fixtures.py", "--output", tmp],
                check=True, capture_output=True, text=True,
            )
            report = json.loads((Path(tmp) / "fixture_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["models"]), set(MODEL_IDS))
            self.assertFalse(report["training_allowed"])
            self.assertFalse(report["owner_approval"])
            self.assertFalse(report["privacy_reviewed"])
            self.assertFalse(report["human_reviewed"])
            self.assertEqual(
                [name for name, row in report["models"].items() if row["fixture_created"]],
                ["chat", "code", "vision", "image", "3d", "research", "document", "safety"],
            )
            self.assertIn("asr", report["models"])
            self.assertFalse(report["models"]["asr"]["fixture_created"])
            self.assertTrue((Path(tmp) / "vision/assets/abstract-1.ppm").is_file())
            self.assertTrue((Path(tmp) / "3d/assets/cloud-1.npy").is_file())
            safety = (Path(tmp) / "safety/records.jsonl").read_text(encoding="utf-8")
            self.assertIn('"decision": null', safety)
            self.assertIn("training_allowed", result.stdout)

            # The existing preparation adapters require all three deterministic splits.
            for model in ("chat", "code", "research", "document", "safety", "vision", "image", "3d"):
                rows = [
                    json.loads(line)
                    for line in (Path(tmp) / model / "records.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                splits = set()
                for row in rows:
                    group_id = row["group_id"]
                    value = int(hashlib.sha256(f"1337:{group_id}".encode()).hexdigest()[:8], 16) % 100
                    splits.add("train" if value < 80 else "validation" if value < 90 else "test")
                self.assertEqual(splits, {"train", "validation", "test"}, model)


    def test_visual_fixtures_run_through_real_preparation_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixtures"
            output = Path(tmp) / "prepared"
            subprocess.run(
                [sys.executable, "scripts/generate_dev_fixtures.py", "--output", str(root)],
                check=True, capture_output=True, text=True,
            )

            for model_id, image_size in (("vision", 224), ("image", 64), ("3d", None)):
                with self.subTest(model_id=model_id):
                    result = prepare_visual_dataset(
                        root / model_id / "records.jsonl",
                        model_id=model_id,
                        output_root=output,
                        license_name="synthetic-development-fixture",
                        permitted_use="smoke-test-only",
                        evidence_uri="local://development-fixture",
                        source_uri="local://development-fixture",
                        vocab_size=256,
                        max_text_length=64,
                        image_size=image_size,
                        point_count=32,
                    )
                    manifest_path = Path(result["manifest"])
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertFalse(result["training_allowed"])
                    self.assertFalse(manifest["training_allowed"])
                    self.assertFalse(manifest["owner_approval"])
                    self.assertFalse(manifest["privacy_reviewed"])
                    self.assertTrue(all(not item["allow_training"] for item in manifest["sources"]))
                    self.assertEqual(manifest["preparation"]["split_counts"], {
                        "train": 1, "validation": 1, "test": 1,
                    })
                    with np.load(result["dataset"], allow_pickle=False) as arrays:
                        self.assertEqual(len(arrays["sample_id"]), 3)
                        self.assertEqual(set(arrays["split"].tolist()), {"train", "validation", "test"})
                        if model_id == "3d":
                            self.assertEqual(arrays["points"].shape, (3, 32, 3))
                        else:
                            self.assertEqual(arrays["images"].shape, (3, 3, image_size, image_size))


    def test_code_and_research_fixtures_run_through_real_text_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixtures"
            output = Path(tmp) / "prepared"
            subprocess.run(
                [sys.executable, "scripts/generate_dev_fixtures.py", "--output", str(root)],
                check=True, capture_output=True, text=True,
            )

            for model_id in ("code", "research"):
                with self.subTest(model_id=model_id):
                    result = prepare_text_dataset(
                        root / model_id / "records.jsonl",
                        model_id=model_id,
                        output_root=output,
                        license_name="synthetic-development-fixture",
                        permitted_use="smoke-test-only",
                        evidence_uri="local://development-fixture",
                        source_uri="local://development-fixture",
                        vocab_size=256,
                        max_length=64,
                    )
                    manifest_path = Path(result["manifest"])
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertFalse(result["training_allowed"])
                    self.assertFalse(manifest["training_allowed"])
                    self.assertFalse(manifest["owner_approval"])
                    self.assertFalse(manifest["privacy_reviewed"])
                    self.assertTrue(all(not item["allow_training"] for item in manifest["sources"]))
                    self.assertEqual(manifest["preparation"]["split_counts"], {
                        "train": 1, "validation": 1, "test": 1,
                    })
                    with np.load(result["dataset"], allow_pickle=False) as arrays:
                        self.assertEqual(len(arrays["sample_id"]), 3)
                        self.assertEqual(set(arrays["split"].tolist()), {"train", "validation", "test"})
                        if model_id == "code":
                            self.assertEqual(arrays["token_ids"].shape, (3, 64))
                        else:
                            self.assertEqual(arrays["query_ids"].shape, (3, 64))
                            self.assertEqual(arrays["passage_ids"].shape, (3, 64))


if __name__ == "__main__":
    unittest.main()
