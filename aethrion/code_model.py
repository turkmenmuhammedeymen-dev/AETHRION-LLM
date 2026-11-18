"""AETHRION-CODE: independent RoPE/SwiGLU decoder trained from random initialization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CodeModelConfig:
    vocab_size: int = 2048
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 256 or self.block_size < 8:
            raise ValueError("Code vocab_size must be >=256 and block_size >=8.")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 8:
            raise ValueError("Code model dimensions must be positive.")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("Code head dimension must be even and divisible by n_head.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Code dropout must be in [0,1).")


class CodeRMSNorm(nn.Module):
    """RMS normalization used only by the independent code architecture."""
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


class CodeRotaryAttention(nn.Module):
    """Causal multi-head attention with rotary positions and separate Q/K/V projections."""
    def __init__(self, config: CodeModelConfig):
        super().__init__()
        self.heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attention_dropout = config.dropout
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inverse_frequencies", inv, persistent=False)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-2)
        positions = torch.arange(length, device=x.device, dtype=self.inverse_frequencies.dtype)
        angles = torch.outer(positions, self.inverse_frequencies)
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = self._rotate(q), self._rotate(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(attended)


class CodeSwiGLU(nn.Module):
    """Gated feed-forward network, distinct from the BASE GELU MLP."""
    def __init__(self, config: CodeModelConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class CodeDecoderBlock(nn.Module):
    def __init__(self, config: CodeModelConfig):
        super().__init__()
        self.attention_norm = CodeRMSNorm(config.n_embd)
        self.attention = CodeRotaryAttention(config)
        self.feedforward_norm = CodeRMSNorm(config.n_embd)
        self.feedforward = CodeSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class AETHRIONCodeLM(nn.Module):
    """Standalone code decoder: its own RoPE attention, RMSNorm and SwiGLU stack."""
    def __init__(self, config: CodeModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([CodeDecoderBlock(config) for _ in range(config.n_layer)])
        self.final_norm = CodeRMSNorm(config.n_embd)
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

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if token_ids.ndim != 2 or token_ids.size(0) < 1 or token_ids.size(1) < 1:
            raise ValueError("token_ids must have shape [batch, sequence] with a non-empty sequence.")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long.")
        if token_ids.size(1) > self.config.block_size:
            raise ValueError("Code sequence exceeds block_size.")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("Code token ID is outside the configured vocabulary.")
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise ValueError("Code targets must be torch.long and match token_ids.")
            supervised = targets != -100
            if not supervised.any():
                raise ValueError("Code targets must contain at least one supervised token.")
            selected = targets[supervised]
            if selected.min().item() < 0 or selected.max().item() >= self.config.vocab_size:
                raise ValueError("Code target ID is outside the configured vocabulary.")
        hidden = self.embedding_dropout(self.token_embedding(token_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            if targets is not None else None
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
        top_k: int | None = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.size(1) < 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a non-empty [batch, sequence] torch.long tensor.")
        if type(max_new_tokens) is not int or max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be >=0 and temperature must be >0.")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be None or a positive integer.")
        if not 0.0 < top_p <= 1.0 or repetition_penalty <= 0:
            raise ValueError("top_p must be in (0,1] and repetition_penalty must be positive.")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("Input token ID is outside the configured vocabulary.")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(token_ids[:, -self.config.block_size:])
                scores = logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    for row in range(token_ids.size(0)):
                        seen = token_ids[row, -128:].unique()
                        values = scores[row, seen]
                        scores[row, seen] = torch.where(
                            values > 0, values / repetition_penalty, values * repetition_penalty
                        )
                if top_k is not None:
                    threshold = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                probabilities = F.softmax(scores, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
                    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
                    remove[:, 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    probabilities = torch.zeros_like(probabilities).scatter(-1, sorted_indices, sorted_probs)
                    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                token_ids = torch.cat((token_ids, torch.multinomial(probabilities, num_samples=1)), dim=1)
            return token_ids
        finally:
            self.train(was_training)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
