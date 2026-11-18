"""Deterministic WAV decoding, resampling and log-mel features for native speech models."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


MAX_WAV_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 120


class AudioFeatureError(ValueError):
    pass


def read_wav(path: str | Path, target_sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size <= 44 or source.stat().st_size > MAX_WAV_BYTES:
        raise AudioFeatureError("WAV file is missing, empty or exceeds 25 MiB.")
    try:
        with wave.open(str(source), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            compression = wav.getcomptype()
            if compression != "NONE" or channels < 1 or channels > 8:
                raise AudioFeatureError("Only uncompressed PCM WAV with 1–8 channels is supported.")
            if width not in {1, 2, 3, 4} or sample_rate < 8000 or sample_rate > 192000:
                raise AudioFeatureError("Unsupported PCM width or sample rate.")
            if frames <= 0 or frames / sample_rate > MAX_SECONDS:
                raise AudioFeatureError("Audio duration must be between 0 and 120 seconds.")
            raw = wav.readframes(frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioFeatureError("Input is not a valid PCM WAV file.") from exc
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = data[:, 0].astype(np.int32) | (data[:, 1].astype(np.int32) << 8) | (data[:, 2].astype(np.int32) << 16)
        values = (values ^ 0x800000) - 0x800000
        samples = values.astype(np.float32) / 8388608.0
    else:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if samples.size != frames * channels:
        raise AudioFeatureError("WAV payload length does not match its header.")
    samples = samples.reshape(frames, channels).mean(axis=1)
    waveform = torch.from_numpy(samples.copy()).float().clamp(-1.0, 1.0)
    if sample_rate != target_sample_rate:
        target_length = max(1, round(waveform.numel() * target_sample_rate / sample_rate))
        waveform = torch.nn.functional.interpolate(
            waveform.view(1, 1, -1), size=target_length, mode="linear", align_corners=False
        ).view(-1)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:
    """Return [frames, n_mels] log-power mel features using native torch operations."""
    if waveform.ndim != 1 or waveform.numel() < n_fft:
        raise AudioFeatureError("waveform must be 1D and contain at least n_fft samples.")
    if not torch.isfinite(waveform).all():
        raise AudioFeatureError("waveform contains non-finite samples.")
    if sample_rate <= 0 or n_mels < 16 or n_fft < 64 or hop_length < 1:
        raise AudioFeatureError("Invalid log-mel parameters.")
    window = torch.hann_window(n_fft, device=waveform.device)
    spectrum = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True
    ).abs().pow(2)
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=waveform.device)
    mel_min = _hz_to_mel(torch.tensor(0.0, device=waveform.device))
    mel_max = _hz_to_mel(torch.tensor(sample_rate / 2, device=waveform.device))
    mel_points = _mel_to_hz(torch.linspace(mel_min, mel_max, n_mels + 2, device=waveform.device))
    lower = mel_points[:-2, None]
    center = mel_points[1:-1, None]
    upper = mel_points[2:, None]
    up = ((frequencies[None, :] - lower) / (center - lower).clamp_min(1e-8)).clamp(0, 1)
    down = ((upper - frequencies[None, :]) / (upper - center).clamp_min(1e-8)).clamp(0, 1)
    filters = torch.minimum(up, down)
    mel = filters @ spectrum
    mel = torch.log(mel.clamp_min(1e-10))
    return mel.transpose(0, 1).contiguous()


def write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> Path:
    if waveform.ndim != 1 or not torch.isfinite(waveform).all():
        raise AudioFeatureError("waveform must be a finite 1D tensor.")
    if waveform.numel() < 1 or waveform.numel() > sample_rate * MAX_SECONDS:
        raise AudioFeatureError("Generated waveform duration is outside 0–120 seconds.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().cpu().clamp(-1, 1).numpy() * 32767.0).round().astype("<i2")
    try:
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
    except (wave.Error, OSError) as exc:
        raise AudioFeatureError("Could not write generated PCM WAV.") from exc
    return target
