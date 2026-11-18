import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from aethrion.audio_features import (
    AudioFeatureError,
    log_mel_spectrogram,
    read_wav,
    write_wav,
)


class NativeAudioFeatureTests(unittest.TestCase):
    def test_pcm_wav_to_log_mel_and_native_wav_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.wav"
            rate = 16000
            samples = (np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 12000).astype("<i2")
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(samples.tobytes())
            waveform, sample_rate = read_wav(source)
            self.assertEqual(sample_rate, rate)
            self.assertEqual(waveform.numel(), rate)
            features = log_mel_spectrogram(waveform, sample_rate, n_mels=40)
            self.assertEqual(features.shape[1], 40)
            self.assertTrue(np.isfinite(features.numpy()).all())
            target = write_wav(root / "generated.wav", waveform)
            self.assertTrue(target.is_file())
            roundtrip, _ = read_wav(target)
            self.assertEqual(roundtrip.numel(), waveform.numel())

    def test_rejects_non_wav_and_oversized_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.wav"
            path.write_bytes(b"not a waveform")
            with self.assertRaises(AudioFeatureError):
                read_wav(path)

    def test_log_mel_rejects_non_finite_waveform(self):
        import torch
        with self.assertRaises(AudioFeatureError):
            log_mel_spectrogram(torch.tensor([float("nan")] * 512))


if __name__ == "__main__":
    unittest.main()
