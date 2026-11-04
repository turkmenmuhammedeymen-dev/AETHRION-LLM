"""Basic held-out next-token evaluation."""
from __future__ import annotations

import argparse
import math

import torch

from aethrion.chat import load_model
from aethrion.train import get_batch, load_tokens
from aethrion.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoint_last.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--data", required=True, help="Held-out UTF-8 text file")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    data = load_tokens(args.data, tokenizer)
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = get_batch(data, 1, model.config.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    print(f"heldout_loss={mean_loss:.4f} perplexity={math.exp(min(mean_loss, 20)):.2f}")


if __name__ == "__main__":
    main()
