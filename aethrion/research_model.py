"""AETHRION-RESEARCH: locally trained query/passage encoder for local corpora."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResearchConfig:
    vocab_size: int = 2048
    max_length: int = 512
    width: int = 256
    layers: int = 4
    heads: int = 8
    embedding_dim: int = 256

    def validate(self):
        if self.vocab_size < 256 or self.max_length < 2:
            raise ValueError("Invalid research vocabulary or context length")
        if min(self.width, self.layers, self.heads, self.embedding_dim) < 1:
            raise ValueError("Research dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class ResearchTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, 4*config.width, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)
        self.projection = nn.Linear(config.width, config.embedding_dim, bias=False)

    def forward(self, token_ids, attention_mask=None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch, tokens] long tensor")
        length = token_ids.size(1)
        if length < 1 or length > self.position.num_embeddings:
            raise ValueError("research text length outside configured bounds")
        mask = None if attention_mask is None else ~attention_mask.bool()
        pos = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.encoder(self.token(token_ids) + self.position(pos), src_key_padding_mask=mask)
        if attention_mask is None:
            pooled = hidden.mean(1)
        else:
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        return F.normalize(self.projection(self.norm(pooled)), dim=-1)


class AETHRIONResearch(nn.Module):
    """Separate query and passage encoders; trained with in-batch positive pairs."""
    def __init__(self, config: ResearchConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.query_encoder = ResearchTextEncoder(config)
        self.passage_encoder = ResearchTextEncoder(config)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        # Hybrid retrieval: learned dense similarity + exact token overlap (generalises to unseen words).
        self.lexical_weight = nn.Parameter(torch.tensor(6.0))

    @staticmethod
    def lexical_overlap(query_ids, passage_ids):
        """Fraction of distinct non-pad query tokens found in each passage -> [queries, passages]."""
        qvalid = query_ids != 0
        hit = (query_ids[:, None, :, None] == passage_ids[None, :, None, :]).any(-1) & qvalid[:, None, :]
        return hit.sum(-1).to(torch.float32) / qvalid.sum(-1).clamp_min(1).to(torch.float32)[:, None]

    def score_matrix(self, query_ids, passage_ids, query_mask=None, passage_mask=None):
        q = self.query_encoder(query_ids, query_mask)
        p = self.passage_encoder(passage_ids, passage_mask)
        dense = self.logit_scale.exp().clamp(max=100) * q @ p.T
        return q, p, dense + self.lexical_weight * self.lexical_overlap(query_ids, passage_ids)

    def forward(self, query_ids, passage_ids, query_mask=None, passage_mask=None):
        if query_ids.size(0) != passage_ids.size(0):
            raise ValueError("query and passage batch sizes must match")
        q, p, logits = self.score_matrix(query_ids, passage_ids, query_mask, passage_mask)
        labels = torch.arange(q.size(0), device=q.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return {"query_embeddings": q, "passage_embeddings": p, "scores": logits, "loss": loss}

    @torch.no_grad()
    def retrieve(self, query_ids, passage_ids, passage_mask=None, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        _, _, scores = self.score_matrix(query_ids, passage_ids, None, passage_mask)
        return torch.topk(scores, min(top_k, scores.size(-1)), dim=-1)

    def config_dict(self):
        return asdict(self.config)
