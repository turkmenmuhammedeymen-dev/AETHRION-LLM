import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from aethrion.specialist_prepare import prepare_text_dataset
from aethrion.specialist_prepare_audio import prepare_audio_dataset
from aethrion.specialist_prepare_visual import prepare_visual_dataset
from aethrion.tokenizer import ByteBPETokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARGS = {
    "license_name": "Synthetic CI fixture only",
    "permitted_use": "Non-production software integration tests only",
    "evidence_uri": "unit-test://synthetic-fixture",
    "source_uri": "unit-test://aethrion-synthetic-fixture",
}


class SyntheticFamilyFixtureTests(unittest.TestCase):
    def _generate(self, output: Path) -> dict:
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "generate_synthetic_family_fixtures.py"),
             "--output", str(output)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        return json.loads((output / "fixture_manifest.json").read_text(encoding="utf-8"))

    def _assert_manifest_closed(self, path: str) -> None:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(manifest["owner_approval"])
        self.assertFalse(manifest["privacy_reviewed"])
        self.assertFalse(manifest["speaker_consent_reviewed"])
        self.assertTrue(all(source["allow_training"] is False for source in manifest["sources"]))

    def test_generator_creates_all_eleven_explicitly_non_training_families(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            manifest = self._generate(output)
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
            for model in ("vision", "image", "3d"):
                rows = [json.loads(line) for line in (output / model / "source.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 30)
                self.assertTrue(all(row["synthetic_fixture"] for row in rows))

    def test_all_eleven_fixture_families_pass_native_preparation_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixtures"
            family = self._generate(root)
            prepared = Path(directory) / "prepared"

            # CHAT reuses a tokenizer artifact, matching the production contract.
            base_tokenizer = Path(directory) / "base-tokenizer.json"
            tokenizer = ByteBPETokenizer()
            tokenizer.train("BASE tokenizer integration fixture", vocab_size=256)
            tokenizer.save(base_tokenizer)

            for model in ("chat", "code", "research", "document", "safety"):
                kwargs = dict(
                    input_path=root / f"{model}.jsonl", model_id=model,
                    output_root=prepared, seed=20261118, max_length=256,
                    **SOURCE_ARGS,
                )
                if model == "chat":
                    kwargs["tokenizer_source_path"] = base_tokenizer
                result = prepare_text_dataset(**kwargs)
                self._assert_manifest_closed(result["manifest"])
                with np.load(result["dataset"], allow_pickle=False) as arrays:
                    self.assertEqual(set(("sample_id", "group_id", "split")).issubset(arrays.files), True)
                    self.assertEqual(len(arrays["sample_id"]), family["record_counts"][model])

            for model in ("vision", "image", "3d"):
                result = prepare_visual_dataset(
                    root / model / "source.jsonl", model_id=model, output_root=prepared,
                    seed=20261118, point_count=64, **SOURCE_ARGS,
                )
                self._assert_manifest_closed(result["manifest"])
                with np.load(result["dataset"], allow_pickle=False) as arrays:
                    self.assertEqual(len(arrays["sample_id"]), 30)

            for model in ("asr", "tts", "vocoder"):
                result = prepare_audio_dataset(
                    root / f"{model}.jsonl", model_id=model, output_root=prepared,
                    seed=20261118, **SOURCE_ARGS,
                )
                self._assert_manifest_closed(result["manifest"])
                with np.load(result["dataset"], allow_pickle=False) as arrays:
                    self.assertEqual(len(arrays["sample_id"]), 30)

            self.assertEqual(len(family["models"]), 11)


if __name__ == "__main__":
    unittest.main()
