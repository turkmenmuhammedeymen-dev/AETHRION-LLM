"""AETHRION-IMAGE: compact text-conditioned pixel diffusion model from random init."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageModelConfig:
    vocab_size: int = 2048
    image_size: int = 64
    text_length: int = 128
    width: int = 64
    time_dim: int = 128
    diffusion_steps: int = 1000

    def validate(self):
        if self.vocab_size < 256 or self.image_size < 16 or self.image_size % 4:
            raise ValueError("Invalid image vocabulary or image size")
        if self.text_length < 1 or self.width < 16 or self.time_dim < 16:
            raise ValueError("Invalid image model dimensions")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >=2")


class ResidualConv(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)

    def forward(self, x, condition):
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class AETHRIONImageGenerator(nn.Module):
    """Small research diffusion U-Net; quality depends entirely on training."""
    def __init__(self, config: ImageModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        w = config.width
        self.text_embedding = nn.Embedding(config.vocab_size, config.time_dim)
        self.text_encoder = nn.GRU(config.time_dim, config.time_dim, batch_first=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim), nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim)
        )
        self.input = nn.Conv2d(3, w, 3, padding=1)
        self.down = nn.Conv2d(w, 2*w, 4, stride=2, padding=1)
        self.down_block = ResidualConv(2*w, config.time_dim)
        self.mid = ResidualConv(2*w, config.time_dim)
        self.up = nn.Conv2d(3*w, w, 3, padding=1)
        self.up_block = ResidualConv(w, config.time_dim)
        self.output = nn.Conv2d(w, 3, 3, padding=1)
        betas = torch.linspace(1e-4, 0.02, config.diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0), persistent=True)

    def _condition(self, token_ids, timesteps):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch, text_length] long tensor")
        if token_ids.size(1) < 1 or token_ids.size(1) > self.config.text_length:
            raise ValueError("text length outside configured bounds")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("text token outside vocabulary")
        _, hidden = self.text_encoder(self.text_embedding(token_ids))
        steps = timesteps.float().unsqueeze(-1) / max(1, self.config.diffusion_steps - 1)
        frequencies = torch.exp(torch.linspace(0, -8, self.config.time_dim // 2, device=steps.device))
        angles = steps * frequencies.unsqueeze(0) * 1000
        time_features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if time_features.size(-1) < self.config.time_dim:
            time_features = F.pad(time_features, (0, self.config.time_dim - time_features.size(-1)))
        return self.time_mlp(hidden[-1] + time_features)

    def forward(self, noisy_images, token_ids, timesteps):
        if noisy_images.ndim != 4 or noisy_images.shape[1] != 3:
            raise ValueError("noisy_images must have shape [batch, 3, H, W]")
        if noisy_images.shape[0] != token_ids.shape[0] or timesteps.shape != (noisy_images.size(0),):
            raise ValueError("image, text and timestep batch sizes must match")
        if tuple(noisy_images.shape[-2:]) != (self.config.image_size, self.config.image_size):
            raise ValueError("image dimensions do not match ImageModelConfig.image_size")
        if timesteps.min().item() < 0 or timesteps.max().item() >= self.config.diffusion_steps:
            raise ValueError("timestep outside diffusion schedule")
        cond = self._condition(token_ids, timesteps)
        x0 = self.input(noisy_images)
        x1 = self.down(F.silu(x0))
        x1 = self.mid(self.down_block(x1), cond)
        x1 = F.interpolate(x1, size=x0.shape[-2:], mode="nearest")
        x = self.up(torch.cat((x0, x1), dim=1))
        return self.output(self.up_block(x, cond))

    def diffusion_loss(self, clean_images, token_ids, timesteps=None, noise=None):
        if clean_images.ndim != 4 or clean_images.size(1) != 3:
            raise ValueError("clean_images must have shape [batch, 3, H, W]")
        batch = clean_images.size(0)
        if timesteps is None:
            timesteps = torch.randint(self.config.diffusion_steps, (batch,), device=clean_images.device)
        if noise is None:
            noise = torch.randn_like(clean_images)
        alpha_bar = self.alpha_bars[timesteps].view(batch, 1, 1, 1)
        noisy = alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise
        prediction = self(noisy, token_ids, timesteps)
        return F.mse_loss(prediction, noise)

    def config_dict(self):
        return asdict(self.config)
