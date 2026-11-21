"""AETHRION-CODE: independently trained causal code model (random initialization)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
from aethrion.model import AETHRIONLM, ModelConfig


@dataclass
class CodeModelConfig:
    vocab_size: int = 2048
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    def validate(self) -> None:
        ModelConfig(**asdict(self)).validate()


class AETHRIONCodeLM(nn.Module):
    """Separate code-specialized causal LM; no pretrained weights are loaded."""
    def __init__(self, config: CodeModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = AETHRIONLM(ModelConfig(**asdict(config)))

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        return self.backbone(token_ids, targets)

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, **sampling):
        return self.backbone.generate(token_ids, max_new_tokens, **sampling)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
