"""Decoder-only causal Transformer language model, initialized from random weights."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 2048
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 256 or self.block_size < 8:
            raise ValueError("vocab_size must be >=256 and block_size >=8")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 8:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.mask[:, :, :length, :length], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(out))

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
                                 nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class AETHRIONLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.ndim != 2 or idx.size(0) < 1 or idx.size(1) < 1:
            raise ValueError("idx must have shape [batch, sequence] with a non-empty sequence")
        if idx.dtype != torch.long:
            raise TypeError("idx must use torch.long token IDs")
        if idx.size(1) > self.config.block_size:
            raise ValueError(f"Sequence length {idx.size(1)} exceeds block_size={self.config.block_size}")
        if idx.min().item() < 0 or idx.max().item() >= self.config.vocab_size:
            raise ValueError("idx contains token IDs outside the configured vocabulary")
        if targets is not None:
            if targets.shape != idx.shape or targets.dtype != torch.long:
                raise ValueError("targets must be torch.long and have the same shape as idx")
            if targets.min().item() < 0 or targets.max().item() >= self.config.vocab_size:
                raise ValueError("targets contain token IDs outside the configured vocabulary")
        batch, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(positions))
        x = self.final_norm(self.blocks(x))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.05) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) < 1 or idx.dtype != torch.long:
            raise ValueError("idx must be a non-empty [batch, sequence] torch.long tensor")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be >= 0 and temperature must be > 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be None or >= 1")
        if not 0.0 < top_p <= 1.0 or repetition_penalty <= 0:
            raise ValueError("top_p must be in (0, 1] and repetition_penalty must be > 0")
        if idx.min().item() < 0 or idx.max().item() >= self.config.vocab_size:
            raise ValueError("idx contains token IDs outside the configured vocabulary")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                logits = logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    for row in range(idx.size(0)):
                        seen = idx[row, -128:].unique()
                        values = logits[row, seen]
                        logits[row, seen] = torch.where(values > 0, values / repetition_penalty,
                                                        values * repetition_penalty)
                if top_k is not None:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
                    remove[:, 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
            return idx
        finally:
            self.train(was_training)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    def config_dict(self) -> dict:
        return asdict(self.config)
