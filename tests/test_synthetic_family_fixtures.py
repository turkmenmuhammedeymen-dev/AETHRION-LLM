import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]


class SyntheticFamilyFixtureTests(unittest.TestCase):
    def test_generator_creates_all_eleven_explicitly_non_training_families(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            subprocess.run(
                [sys.executable, str(ROOT / "tools" / "generate_synthetic_family_fixtures.py"),
                 "--output", str(output)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            manifest = json.loads((output / "fixture_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["models"]), 11)
            self.assertTrue(manifest["not_for_training"])
            self.assertFalse(manifest["training_allowed"])
            self.assertFalse(manifest["owner_approval"])
            self.assertFalse(manifest["human_reviewed"])
            self.assertFalse(manifest["speaker_consent_reviewed"])
            self.assertEqual(manifest["record_counts"]["safety"], 225)

            for model in ("chat", "code", "research", "document", "safety"):
                rows = [json.loads(line) for line in (output / f"{model}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(rows)
                self.assertTrue(all(row["synthetic_fixture"] for row in rows))
                self.assertTrue(all(not row["production_training_approved"] for row in rows))

            for model in ("asr", "tts", "vocoder"):
                rows = [json.loads(line) for line in (output / f"{model}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 30)
                self.assertTrue(all((output / row["wav_path"]).is_file() for row in rows))
                with wave.open(str(output / rows[0]["wav_path"]), "rb") as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.getnchannels(), 1)

            for model in ("vision", "image", "3d"):
                rows = [json.loads(line) for line in (output / model / "source.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 30)
                self.assertTrue(all(row["synthetic_fixture"] for row in rows))


if __name__ == "__main__":
    unittest.main()
