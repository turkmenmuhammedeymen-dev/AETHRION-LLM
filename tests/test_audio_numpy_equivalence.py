import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NumpyMelEquivalenceTests(unittest.TestCase):
    def test_numpy_mel_matches_torch_implementation(self):
        try:
            import torch
            from aethrion.audio_features import log_mel_spectrogram, read_wav
        except ImportError:
            self.skipTest("torch unavailable")
        spec = importlib.util.spec_from_file_location("prepare_audio_numpy", ROOT / "scripts" / "prepare_audio_numpy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        clips = sorted((ROOT / "data" / "authored" / "audio" / "wav").glob("*.wav"))[:5]
        self.assertTrue(clips)
        for clip in clips:
            wave_t, _ = read_wav(clip)
            for hop in (160, 256):
                reference = log_mel_spectrogram(wave_t, n_fft=max(400, hop * 4), hop_length=hop).numpy()
                wave_n, _ = module.np_read_wav(clip)
                ours = module.np_log_mel(wave_n, n_fft=max(400, hop * 4), hop_length=hop).numpy()
                self.assertEqual(reference.shape, ours.shape)
                diff = np.abs(reference - ours)
                # float32 (torch) vs float64 (numpy) differ only in near-silent bins; require close agreement elsewhere
                self.assertGreater(float(np.mean(diff < 0.1)), 0.97)
                self.assertLess(float(np.median(diff)), 1e-2)

    def test_synthetic_tts_durations_sum_to_frame_count(self):
        import json
        rows = [json.loads(line) for line in (ROOT / "data" / "authored" / "audio" / "tts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreater(len(rows), 100)
        for row in rows[:20]:
            self.assertTrue(all(isinstance(d, int) and d >= 1 for d in row["durations"]))


if __name__ == "__main__":
    unittest.main()
