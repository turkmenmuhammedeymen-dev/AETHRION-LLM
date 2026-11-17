"""AETHRION-ASR: random-init acoustic encoder with CTC transcription objective."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ASRConfig:
    vocab_size: int = 2048
    n_mels: int = 80
    width: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    blank_id: int = 0

    def validate(self):
        if self.vocab_size < 2 or self.n_mels < 16:
            raise ValueError("Invalid ASR vocabulary or mel feature dimension")
        if min(self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("ASR width must be divisible by positive head count")
        if not 0 <= self.blank_id < self.vocab_size or not 0 <= self.dropout < 1:
            raise ValueError("Invalid CTC blank id or dropout")


class AETHRIONASR(nn.Module):
    """Maps log-mel frames to token distributions; trained on consented speech."""
    def __init__(self, config: ASRConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.frontend = nn.Sequential(
            nn.Conv1d(config.n_mels, config.width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(config.width, config.width, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4*config.width, config.dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)
        self.classifier = nn.Linear(config.width, config.vocab_size)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor | None = None):
        if features.ndim != 3 or features.size(-1) != self.config.n_mels:
            raise ValueError("features must have shape [batch, frames, n_mels]")
        if features.size(1) < 4:
            raise ValueError("ASR requires at least four acoustic frames")
        hidden = self.frontend(features.transpose(1, 2)).transpose(1, 2)
        output_lengths = None
        if input_lengths is not None:
            if input_lengths.ndim != 1 or input_lengths.numel() != features.size(0):
                raise ValueError("input_lengths must have one entry per batch item")
            output_lengths = torch.div(input_lengths + 3, 4, rounding_mode="floor").clamp(min=1, max=hidden.size(1))
            positions = torch.arange(hidden.size(1), device=hidden.device)[None, :]
            padding_mask = positions >= output_lengths[:, None]
        else:
            padding_mask = None
            output_lengths = torch.full((features.size(0),), hidden.size(1), device=hidden.device, dtype=torch.long)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        log_probs = F.log_softmax(self.classifier(self.norm(hidden)), dim=-1)
        return log_probs.transpose(0, 1), output_lengths

    def ctc_loss(self, features, targets, input_lengths, target_lengths):
        log_probs, encoded_lengths = self(features, input_lengths)
        if targets.dtype != torch.long or targets.ndim != 1:
            raise ValueError("CTC targets must be a flat torch.long tensor")
        if target_lengths.shape != input_lengths.shape:
            raise ValueError("target_lengths must match input_lengths shape")
        loss = F.ctc_loss(
            log_probs, targets, encoded_lengths, target_lengths,
            blank=self.config.blank_id, reduction="mean", zero_infinity=True
        )
        return loss

    def config_dict(self):
        return asdict(self.config)
