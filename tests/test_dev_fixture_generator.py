import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
