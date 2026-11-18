"""Synthetic end-to-end check; does not download external models or datasets."""
import tempfile
from pathlib import Path
import torch
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer

def main():
    sample = ("AETHRION bağımsız dil modeli deneme metni. " * 80) + "\n"
    tokenizer = ByteBPETokenizer()
    # Keep byte tokens unmerged so this test always has enough sequence positions.
    # Learned-merge behavior is covered independently by tokenizer unit tests.
    tokenizer.train(sample, vocab_size=256, min_frequency=2)
    ids = tokenizer.encode(sample)
    if len(ids) < 64:
        raise RuntimeError("Synthetic smoke corpus unexpectedly short.")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=1,
                         n_head=2, n_embd=32, dropout=0.0)
    model = AETHRIONLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.tensor([ids[:32]]), torch.tensor([ids[1:33]])
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    generated = model.generate(x[:, :8], max_new_tokens=2, top_k=5)
    if generated.shape != (1, 10):
        raise RuntimeError("Unexpected generation shape")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tokenizer.json"
        tokenizer.save(path)
        loaded = ByteBPETokenizer.load(path)
        if loaded.decode(loaded.encode("Türkçe deneme")) != "Türkçe deneme":
            raise RuntimeError("Tokenizer round-trip failed")
    print(f"Smoke test passed | loss={loss.item():.4f} | params={model.parameter_count():,}")

if __name__ == "__main__":
    main()
