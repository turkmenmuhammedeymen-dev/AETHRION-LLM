"""AETHRION-DOC: text/layout encoder for token tagging and extractive QA."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DocumentConfig:
    vocab_size: int = 4096
    layout_vocab_size: int = 32
    max_length: int = 512
    width: int = 256
    layers: int = 6
    heads: int = 8
    tag_classes: int = 32

    def validate(self):
        if self.vocab_size < 256 or min(self.layout_vocab_size, self.max_length, self.tag_classes) < 1:
            raise ValueError("Invalid document vocabulary or output dimensions")
        if min(self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("Document width must be divisible by positive head count")


class AETHRIONDocument(nn.Module):
    """Consumes OCR text, normalized boxes and layout tags from local preprocessing."""
    def __init__(self, config: DocumentConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.layout = nn.Embedding(config.layout_vocab_size, config.width)
        self.box_projection = nn.Sequential(nn.Linear(4, config.width), nn.GELU(), nn.Linear(config.width, config.width))
        self.position = nn.Embedding(config.max_length, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, 4*config.width, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)
        self.tag_head = nn.Linear(config.width, config.tag_classes)
        self.qa_head = nn.Linear(config.width, 2)

    def forward(self, token_ids, boxes, layout_ids, attention_mask=None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be [batch,tokens] long tensor")
        if boxes.shape != (*token_ids.shape, 4):
            raise ValueError("boxes must have shape [batch,tokens,4]")
        if layout_ids.shape != token_ids.shape or layout_ids.dtype != torch.long:
            raise ValueError("layout_ids must match token_ids and be long")
        if token_ids.size(1) < 1 or token_ids.size(1) > self.config.max_length:
            raise ValueError("document sequence length outside configured bounds")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("document token outside vocabulary")
        if layout_ids.min().item() < 0 or layout_ids.max().item() >= self.config.layout_vocab_size:
            raise ValueError("layout id outside vocabulary")
        if not torch.isfinite(boxes).all() or boxes.min().item() < 0 or boxes.max().item() > 1:
            raise ValueError("boxes must contain finite normalized coordinates in [0,1]")
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        hidden = self.token(token_ids) + self.layout(layout_ids) + self.box_projection(boxes) + self.position(positions)
        mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=mask))
        return {"tag_logits": self.tag_head(hidden), "qa_logits": self.qa_head(hidden), "hidden": hidden}

    def multitask_loss(self, outputs, tag_labels=None, start_positions=None, end_positions=None,
                       attention_mask=None):
        losses = []
        if tag_labels is not None:
            tags = F.cross_entropy(outputs["tag_logits"].reshape(-1, self.config.tag_classes),
                                   tag_labels.reshape(-1), ignore_index=-100)
            losses.append(tags)
        if start_positions is not None or end_positions is not None:
            if start_positions is None or end_positions is None:
                raise ValueError("both QA start and end positions are required")
            start_loss = F.cross_entropy(outputs["qa_logits"][..., 0], start_positions, ignore_index=-100)
            end_loss = F.cross_entropy(outputs["qa_logits"][..., 1], end_positions, ignore_index=-100)
            losses.append((start_loss + end_loss) / 2)
        if not losses:
            raise ValueError("at least one document supervision target is required")
        return torch.stack(losses).mean()

    def config_dict(self):
        return asdict(self.config)
