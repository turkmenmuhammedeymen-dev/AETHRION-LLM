"""Local terminal chat for a trained AETHRION checkpoint."""
from __future__ import annotations

import argparse

import torch

from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer


def load_model(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**state["model_config"])
    model = AETHRIONLM(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoint_last.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    print("AETHRION-0.1 terminal chat. Çıkmak için /exit yaz.")
    while True:
        prompt = input("\nSen: ").strip()
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        ids = tokenizer.encode(prompt)
        if not ids:
            continue
        context = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model.generate(context, args.max_new_tokens)
        print("AETHRION:", tokenizer.decode(result[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
