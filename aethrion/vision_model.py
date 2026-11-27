"""AETHRION-VISION: random-init image/text dual encoder with contrastive loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    vocab_size: int = 2048
    max_text_length: int = 128
    width: int = 256
    layers: int = 6
    heads: int = 8
    projection_dim: int = 256
    dropout: float = 0.1

    def validate(self):
        if self.image_size < 32 or self.patch_size < 4 or self.image_size % self.patch_size:
            raise ValueError("image_size must be >=32 and divisible by patch_size")
        if self.vocab_size < 256 or self.max_text_length < 2:
            raise ValueError("Invalid vocabulary or text context length")
        if min(self.width, self.layers, self.heads, self.projection_dim) < 1:
            raise ValueError("Vision dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class AETHRIONVision(nn.Module):
    """CLIP-style visual/text encoders trained only on approved local pairs."""
    def __init__(self, config: VisionConfig):
        super().__init__()
        config.validate()
        self.config = config
        grid = config.image_size // config.patch_size
        count = grid * grid
        self.patch = nn.Conv2d(3, config.width, config.patch_size, stride=config.patch_size)
        self.image_cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.image_pos = nn.Parameter(torch.zeros(1, count + 1, config.width))
        image_layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4 * config.width, config.dropout,
            batch_first=True, norm_first=True
        )
        self.image_encoder = nn.TransformerEncoder(image_layer, config.layers)
        self.image_norm = nn.LayerNorm(config.width)
        self.image_projection = nn.Linear(config.width, config.projection_dim, bias=False)

        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.text_pos = nn.Embedding(config.max_text_length, config.width)
        text_layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4 * config.width, config.dropout,
            batch_first=True, norm_first=True
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, config.layers)
        self.text_norm = nn.LayerNorm(config.width)
        self.text_projection = nn.Linear(config.width, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592600369))
        nn.init.normal_(self.image_cls, std=0.02)
        nn.init.normal_(self.image_pos, std=0.02)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if tuple(images.shape[-2:]) != (self.config.image_size, self.config.image_size):
            raise ValueError("image dimensions do not match VisionConfig.image_size")
        patches = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.image_cls.expand(images.size(0), -1, -1)
        features = self.image_encoder(torch.cat((cls, patches), dim=1) + self.image_pos)
        return F.normalize(self.image_projection(self.image_norm(features[:, 0])), dim=-1)

    def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a [batch, sequence] torch.long tensor")
        batch, length = token_ids.shape
        if length < 1 or length > self.config.max_text_length:
            raise ValueError("text sequence length is outside configured bounds")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("text token id outside vocabulary")
        padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask shape must match token_ids")
            padding_mask = ~attention_mask.bool()
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.text_encoder(
            self.token_embedding(token_ids) + self.text_pos(positions),
            src_key_padding_mask=padding_mask
        )
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(self.text_projection(self.text_norm(pooled)), dim=-1)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=20)  # cap overconfidence on tiny datasets
        logits = scale * image_features @ text_features.transpose(0, 1)
        labels = torch.arange(images.size(0), device=images.device)
        if token_ids.size(0) != images.size(0):
            raise ValueError("image and text batch sizes must match")
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return {"image_features": image_features, "text_features": text_features,
                "logits_per_image": logits, "loss": loss}

    def config_dict(self):
        return asdict(self.config)
