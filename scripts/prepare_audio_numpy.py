"""Run aethrion.specialist_prepare_audio WITHOUT torch (numpy log-mel identical in definition to aethrion.audio_features).
For environments lacking torch. Equivalence with the torch implementation is asserted in tests/test_audio_numpy_equivalence.py.
Usage: PYTHONPATH=. python3 scripts/prepare_audio_numpy.py <same CLI arguments as aethrion.specialist_prepare_audio>"""
import sys, types, wave
import numpy as np

class _T:  # minimal tensor stand-in exposing what the preparation tool touches
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def numel(self): return self.a.size
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return self.a[i]
    @property
    def ndim(self): return self.a.ndim

def np_read_wav(path, target_sample_rate=16000):
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == target_sample_rate
        s = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return _T(np.clip(s, -1, 1)), target_sample_rate

def np_log_mel(waveform, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    x = np.asarray(getattr(waveform, "a", waveform), dtype=np.float32)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    p = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    nf = 1 + (len(p) - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(nf)[:, None]
    spec = np.abs(np.fft.rfft(p[idx] * win, axis=1)) ** 2  # [frames, bins]
    fr = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    h2m = lambda v: 2595.0 * np.log10(1.0 + v / 700.0); m2h = lambda v: 700.0 * (10 ** (v / 2595.0) - 1.0)
    pts = m2h(np.linspace(h2m(0.0), h2m(sample_rate / 2), n_mels + 2))
    lo, ce, up = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    filt = np.minimum(np.clip((fr[None] - lo) / np.maximum(ce - lo, 1e-8), 0, 1), np.clip((up - fr[None]) / np.maximum(up - ce, 1e-8), 0, 1))
    mel = np.log(np.maximum(filt @ spec.T, 1e-10))  # [mels, frames]
    return _T(mel.T.astype(np.float32))

if __name__ == "__main__":
    stub = types.ModuleType("torch"); stub.Tensor = _T; sys.modules.setdefault("torch", stub)
    import aethrion.specialist_prepare_audio as spa
    spa.read_wav, spa.log_mel_spectrogram = np_read_wav, np_log_mel
    spa.main()
