import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


class AllModelSyntheticFixtureTests(unittest.TestCase):
    def test_all_eleven_model_fixtures_are_labeled_and_unapproved(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixtures"
            completed = subprocess.run(
                [sys.executable, str(root / "scripts" / "build_all_model_synthetic_fixtures.py"), "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(completed.stdout)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            expected = {"chat", "code", "vision", "image", "asr", "tts", "vocoder", "3d", "research", "document", "safety"}
            self.assertEqual(set(manifest["models"]), expected)
            self.assertEqual(result["model_count"], 11)
            self.assertEqual(manifest["origin"], "ai_generated_synthetic")
            self.assertIs(manifest["training_allowed"], False)
            self.assertIs(manifest["owner_approval"], False)
            self.assertIs(manifest["rights_holder_attestation"], False)
            self.assertIs(manifest["production_training_eligible"], False)
            for model in expected:
                self.assertEqual(manifest["models"][model]["origin"], "ai_generated_synthetic")
                self.assertIs(manifest["models"][model]["training_allowed"], False)
                records = [json.loads(line) for line in (output / model / "records.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 4)
                self.assertEqual(len({row["group_id"] for row in records}), 4)
            for model in ("vision", "image"):
                self.assertTrue((output / model / "assets" / f"{model}-0.ppm").is_file())
            for model in ("asr", "tts", "vocoder"):
                with wave.open(str(output / model / "assets" / f"{model}-0.wav"), "rb") as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.getnchannels(), 1)
            points = json.loads((output / "3d" / "assets" / "shape-0.json").read_text(encoding="utf-8"))
            self.assertEqual(len(points["points"]), 16)
            safety = [json.loads(line)["decision"] for line in (output / "safety" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(safety), {"ALLOW", "REVIEW", "BLOCK"})
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, entry["bytes"])


if __name__ == "__main__":
    unittest.main()
