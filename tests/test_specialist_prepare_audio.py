import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from aethrion.specialist_prepare_audio import prepare_audio_dataset


class AudioPreparationTests(unittest.TestCase):
    def _wav(self, path: Path):
        samples = (np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 220 / 16000)) * 12000)
        pcm = samples.astype("<i2").tobytes()
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(pcm)

    def _rows(self, root: Path, *, transcript=False, durations=False):
        audio = root / "sample.wav"
        self._wav(audio)
        rows = []
        for index in range(30):
            row = {"id": f"speech-{index}", "wav_path": "sample.wav"}
            if transcript:
                row["transcript"] = "a"
            if durations:
                row["durations"] = [101]
            rows.append(row)
        source = root / "audio.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return source

    def _kwargs(self, root: Path):
        return {
            "output_root": root / "prepared",
            "license_name": "Unit test only",
            "permitted_use": "Temporary CI fixture only",
            "evidence_uri": "unit-test://audio-source",
            "source_uri": "unit-test://audio-fixture",
        }

    def test_asr_wav_to_logmel_ctc_arrays_keeps_approval_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._rows(root, transcript=True)
            result = prepare_audio_dataset(source, model_id="asr", **self._kwargs(root))
            with np.load(result["dataset"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["features"].shape[2], 80)
                self.assertEqual(arrays["targets"].shape[1], 1)
                self.assertTrue(np.all(arrays["input_lengths"] >= 4))
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_allowed"])
            self.assertFalse(manifest["speaker_consent_reviewed"])
            self.assertTrue(all(not item["speaker_consent_verified"] for item in manifest["sources"]))

    def test_tts_wav_requires_token_level_duration_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._rows(root, transcript=True, durations=True)
            result = prepare_audio_dataset(source, model_id="tts", **self._kwargs(root))
            with np.load(result["dataset"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["token_ids"].shape, arrays["durations"].shape)
                self.assertTrue(np.all(arrays["token_mask"].sum(axis=1) == 1))
                self.assertTrue(np.all(arrays["durations"].sum(axis=1) == arrays["frame_lengths"]))
                self.assertTrue(np.all(arrays["frame_mask"].sum(axis=1) == arrays["frame_lengths"]))

    def test_vocoder_wav_uses_model_hop_and_waveform_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._rows(root)
            result = prepare_audio_dataset(source, model_id="vocoder", **self._kwargs(root))
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["preparation"]["hop_length"], 256)
            with np.load(result["dataset"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["target_waveforms"].shape[1], arrays["target_mels"].shape[1] * 256)
                self.assertTrue(np.all(arrays["waveform_mask"].sum(axis=1) == 16000))


if __name__ == "__main__":
    unittest.main()
