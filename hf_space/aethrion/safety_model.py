"""AETHRION-SAFE: independently trained text policy classifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SafetyConfig:
    vocab_size: int = 4096
    max_length: int = 512
    width: int = 256
    layers: int = 4
    heads: int = 8
    reason_classes: int = 16
    decisions: int = 3

    def validate(self):
        if self.vocab_size < 256 or self.max_length < 2 or self.reason_classes < 1:
            raise ValueError("Invalid safety model vocabulary or dimensions")
        if self.decisions != 3:
            raise ValueError("Safety decision head must expose ALLOW, REVIEW and BLOCK")
        if min(self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("Safety width must be divisible by positive head count")


class AETHRIONSafe(nn.Module):
    """Returns calibrated candidates only after external calibration/evaluation."""
    DECISIONS = ("ALLOW", "REVIEW", "BLOCK")

    def __init__(self, config: SafetyConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, 4*config.width, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)
        self.decision_head = nn.Linear(config.width, config.decisions)
        self.reason_head = nn.Linear(config.width, config.reason_classes)

    def forward(self, token_ids, attention_mask=None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch,tokens] long tensor")
        length = token_ids.size(1)
        if length < 1 or length > self.config.max_length:
            raise ValueError("safety text length outside configured bounds")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("safety token outside vocabulary")
        pos = torch.arange(length, device=token_ids.device).unsqueeze(0)
        mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.encoder(self.token(token_ids) + self.position(pos), src_key_padding_mask=mask)
        if attention_mask is None:
            pooled = hidden.mean(1)
        else:
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        pooled = self.norm(pooled)
        return {"decision_logits": self.decision_head(pooled),
                "reason_logits": self.reason_head(pooled)}

    def loss(self, outputs, decision_labels, reason_labels=None):
        decision_loss = F.cross_entropy(outputs["decision_logits"], decision_labels)
        if reason_labels is None:
            return decision_loss
        reason_loss = F.binary_cross_entropy_with_logits(outputs["reason_logits"], reason_labels.float())
        return decision_loss + reason_loss

    @torch.no_grad()
    def predict_candidate(self, token_ids, attention_mask=None, review_threshold=0.65):
        if not 0.0 < review_threshold < 1.0:
            raise ValueError("review_threshold must be in (0,1)")
        outputs = self(token_ids, attention_mask)
        probs = outputs["decision_logits"].softmax(-1)
        confidence, indices = probs.max(-1)
        # Low-confidence predictions abstain to REVIEW; this is not a release-ready safety gate.
        review = confidence < review_threshold
        indices = torch.where(review, torch.ones_like(indices), indices)
        names = [self.DECISIONS[index] for index in indices.tolist()]
        return {"decision": names, "confidence": confidence, "reason_scores": outputs["reason_logits"].sigmoid()}

    def config_dict(self):
        return asdict(self.config)
