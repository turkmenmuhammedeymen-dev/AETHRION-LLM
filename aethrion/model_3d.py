"""AETHRION-3D: text-conditioned point-cloud diffusion from random initialization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Model3DConfig:
    vocab_size: int = 2048
    width: int = 256
    points: int = 1024
    layers: int = 6
    heads: int = 8
    diffusion_steps: int = 1000

    def validate(self):
        if self.vocab_size < 256 or self.points < 16:
            raise ValueError("Invalid 3D vocabulary or point count")
        if min(self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("3D width must be divisible by positive head count")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >=2")


class AETHRION3D(nn.Module):
    """Denoises normalized point clouds conditioned on text; not a mesh renderer."""
    def __init__(self, config: Model3DConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.text_embedding = nn.Embedding(config.vocab_size, config.width)
        text_layer = nn.TransformerEncoderLayer(config.width, config.heads, 4*config.width, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(text_layer, config.layers)
        self.point_input = nn.Linear(3, config.width)
        self.time_embedding = nn.Embedding(config.diffusion_steps, config.width)
        point_layer = nn.TransformerEncoderLayer(config.width, config.heads, 4*config.width, batch_first=True)
        self.point_encoder = nn.TransformerEncoder(point_layer, config.layers)
        self.condition = nn.Linear(config.width, config.width)
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 3))
        betas = torch.linspace(1e-4, 0.02, config.diffusion_steps)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def forward(self, noisy_points, token_ids, timesteps):
        if noisy_points.ndim != 3 or noisy_points.size(-1) != 3:
            raise ValueError("noisy_points must have shape [batch, points, 3]")
        if noisy_points.size(1) != self.config.points:
            raise ValueError("point count does not match Model3DConfig")
        if token_ids.ndim != 2 or token_ids.dtype != torch.long or token_ids.size(0) != noisy_points.size(0):
            raise ValueError("token_ids must be [batch, text] and match point batch")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("3D condition token outside vocabulary")
        if timesteps.shape != (noisy_points.size(0),) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a long tensor with one entry per sample")
        if timesteps.min().item() < 0 or timesteps.max().item() >= self.config.diffusion_steps:
            raise ValueError("timestep outside diffusion schedule")
        text = self.text_encoder(self.text_embedding(token_ids)).mean(dim=1)
        cond = self.condition(text + self.time_embedding(timesteps))
        points = self.point_input(noisy_points) + cond.unsqueeze(1)
        return self.output(self.point_encoder(points))

    def diffusion_loss(self, clean_points, token_ids, timesteps=None, noise=None):
        if clean_points.ndim != 3 or clean_points.shape[1:] != (self.config.points, 3):
            raise ValueError("clean_points must have shape [batch, configured_points, 3]")
        batch = clean_points.size(0)
        if timesteps is None:
            timesteps = torch.randint(self.config.diffusion_steps, (batch,), device=clean_points.device)
        if noise is None:
            noise = torch.randn_like(clean_points)
        alpha = self.alpha_bars[timesteps].view(batch, 1, 1)
        noisy = alpha.sqrt() * clean_points + (1-alpha).sqrt() * noise
        return F.mse_loss(self(noisy, token_ids, timesteps), noise)

    def config_dict(self):
        return asdict(self.config)


def point_cloud_to_obj(points: torch.Tensor) -> str:
    """Serialize a bounded [N,3] point set as OBJ vertices (not a watertight mesh)."""
    if points.ndim != 2 or points.size(1) != 3 or not 1 <= points.size(0) <= 100_000:
        raise ValueError("points must have shape [N,3] with 1..100000 vertices")
    if not torch.isfinite(points).all():
        raise ValueError("point cloud contains non-finite coordinates")
    rows = ["# AETHRION point cloud; vertices only, no faces"]
    rows.extend(f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in points.detach().cpu().tolist())
    return "\n".join(rows) + "\n"
