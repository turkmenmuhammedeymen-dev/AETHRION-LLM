"""AETHRION-TTS: independently trained acoustic predictor and neural vocoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTSConfig:
    vocab_size: int = 2048
    mel_bins: int = 80
    width: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1

    def validate(self):
        if self.vocab_size < 256 or self.mel_bins < 16:
            raise ValueError("Invalid TTS vocabulary or mel dimension")
        if min(self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("TTS width must be divisible by positive head count")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class AETHRIONTTS(nn.Module):
    """Predicts mel frames from transcript tokens and frame-to-token alignment."""
    def __init__(self, config: TTSConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4*config.width, config.dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.duration_predictor = nn.Sequential(
            nn.Linear(config.width, config.width), nn.GELU(),
            nn.Linear(config.width, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(config.width, config.width, 5, padding=2),
            nn.GELU(), nn.Conv1d(config.width, config.width, 5, padding=2),
            nn.GELU(), nn.Conv1d(config.width, config.mel_bins, 1)
        )

    def forward(self, token_ids: torch.Tensor, frame_to_token: torch.Tensor):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch, tokens] long tensor")
        if frame_to_token.ndim != 2 or frame_to_token.dtype != torch.long:
            raise ValueError("frame_to_token must be [batch, frames] long tensor")
        if token_ids.size(0) != frame_to_token.size(0) or token_ids.size(1) < 1:
            raise ValueError("TTS batch dimensions are invalid")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("TTS token outside vocabulary")
        if frame_to_token.min().item() < 0 or frame_to_token.max().item() >= token_ids.size(1):
            raise ValueError("frame_to_token references a missing input token")
        encoded = self.encoder(self.embedding(token_ids))
        aligned = encoded.gather(1, frame_to_token.unsqueeze(-1).expand(-1, -1, encoded.size(-1)))
        return self.decoder(aligned.transpose(1, 2)).transpose(1, 2)

    def predict_durations(self, token_ids: torch.Tensor, min_frames: int = 1,
                          max_frames: int = 200) -> torch.Tensor:
        if min_frames < 1 or max_frames < min_frames:
            raise ValueError("Invalid duration prediction bounds")
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch,tokens] long tensor")
        encoded = self.encoder(self.embedding(token_ids))
        predicted = torch.expm1(self.duration_predictor(encoded).squeeze(-1)).round()
        return predicted.clamp(min=min_frames, max=max_frames).long()

    @staticmethod
    def durations_to_alignment(durations: torch.Tensor) -> torch.Tensor:
        if durations.ndim != 2 or durations.dtype != torch.long or durations.min().item() < 1:
            raise ValueError("durations must be a positive [batch,tokens] long tensor")
        rows = []
        for row in durations:
            rows.append(torch.repeat_interleave(
                torch.arange(row.numel(), device=row.device), row
            ))
        lengths = torch.tensor([row.numel() for row in rows], dtype=torch.long, device=durations.device)
        max_length = int(lengths.max().item())
        padded = [F.pad(row, (0, max_length - row.numel()), value=0) for row in rows]
        return torch.stack(padded), lengths

    def duration_loss(self, token_ids, target_durations, token_mask=None):
        if target_durations.shape != token_ids.shape or target_durations.dtype != torch.long:
            raise ValueError("target_durations must be a long tensor matching token_ids")
        if token_mask is None:
            token_mask = torch.ones_like(target_durations, dtype=torch.bool)
        if token_mask.shape != token_ids.shape or not token_mask.bool().any():
            raise ValueError("token_mask must identify at least one real token per batch")
        if target_durations[token_mask.bool()].min().item() < 1 or torch.any(target_durations[~token_mask.bool()] != 0):
            raise ValueError("real-token durations must be positive and padded durations must be zero")
        encoded = self.encoder(self.embedding(token_ids))
        predicted_log = self.duration_predictor(encoded).squeeze(-1)
        losses = F.smooth_l1_loss(predicted_log, torch.log1p(target_durations.float()), reduction="none")
        mask = token_mask.to(losses.dtype)
        return (losses * mask).sum() / mask.sum().clamp_min(1)

    def acoustic_loss(self, token_ids, frame_to_token, target_mels, frame_mask=None):
        predicted = self(token_ids, frame_to_token)
        if predicted.shape != target_mels.shape:
            raise ValueError("predicted and target mel spectrogram shapes must match")
        loss = F.l1_loss(predicted, target_mels, reduction="none")
        if frame_mask is not None:
            if frame_mask.shape != target_mels.shape[:2]:
                raise ValueError("frame_mask shape must match [batch, frames]")
            loss = loss * frame_mask.unsqueeze(-1).to(loss.dtype)
            return loss.sum() / (frame_mask.sum() * target_mels.size(-1)).clamp_min(1)
        return loss.mean()

    def config_dict(self):
        return asdict(self.config)


class AETHRIONVocoder(nn.Module):
    """Small learned mel-to-waveform upsampler; requires paired waveform supervision."""
    def __init__(self, mel_bins: int = 80, channels: int = 128, hop_length: int = 256):
        super().__init__()
        if mel_bins < 16 or channels < 16 or hop_length < 2 or hop_length & (hop_length - 1):
            raise ValueError("Invalid vocoder dimensions; hop_length must be a power of two")
        self.mel_bins, self.hop_length = mel_bins, hop_length
        stages = hop_length.bit_length() - 1
        layers = [nn.Conv1d(mel_bins, channels, 7, padding=3), nn.LeakyReLU(0.1)]
        width = channels
        for _ in range(stages):
            next_width = max(16, width // 2)
            layers.extend([
                nn.ConvTranspose1d(width, next_width, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(next_width, next_width, 3, padding=1),
                nn.LeakyReLU(0.1),
            ])
            width = next_width
        layers.extend([nn.Conv1d(width, 1, 7, padding=3), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, mel):
        if mel.ndim != 3 or mel.size(1) != self.mel_bins:
            raise ValueError("mel must have shape [batch, mel_bins, frames]")
        return self.network(mel).squeeze(1)

    @staticmethod
    def waveform_loss(predicted, target, sample_mask=None):
        if predicted.shape != target.shape:
            raise ValueError("predicted and target waveform shapes must match")
        error = (predicted - target).abs()
        if sample_mask is not None:
            if sample_mask.shape != target.shape:
                raise ValueError("sample_mask must match waveform shape")
            error = error * sample_mask.to(error.dtype)
            return error.sum() / sample_mask.sum().clamp_min(1)
        return error.mean()
