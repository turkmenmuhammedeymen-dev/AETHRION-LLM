"""A compact byte-level BPE tokenizer; all merges are learned from the supplied corpus."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self._rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @staticmethod
    def _bytes(text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def train(self, text: str, vocab_size: int = 2048, min_frequency: int = 2) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level tokenization")
        sequences = [self._bytes(line) for line in text.splitlines() if line.strip()]
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            updated = []
            for seq in sequences:
                out: list[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                updated.append(out)
            sequences = updated
        self.merges = merges
        self._rank = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_size = 256 + len(merges)

    def encode(self, text: str) -> list[int]:
        seq = self._bytes(text)
        while len(seq) > 1:
            best_pair = None
            best_rank = len(self.merges) + 1
            for pair in zip(seq, seq[1:]):
                rank = self._rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            new_id = 256 + best_rank
            out: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        return seq

    def decode(self, ids: list[int]) -> str:
        pieces: list[bytes] = [bytes([i]) if 0 <= i < 256 else b"" for i in ids]
        for rank, (left, right) in enumerate(self.merges):
            token_id = 256 + rank
            # Reconstruct merged token bytes from the already reconstructed token table.
            if token_id < len(pieces):
                continue
            pieces.append(pieces[left] + pieces[right])
        raw = b"".join(pieces[i] if 0 <= i < len(pieces) else b"" for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "aethrion-byte-bpe-v1", "merges": self.merges},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "aethrion-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(pair) for pair in payload["merges"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AETHRION byte-level BPE tokenizer")
    parser.add_argument("--train", required=True, help="UTF-8 corpus text file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    text = Path(args.train).read_text(encoding="utf-8")
    tokenizer = ByteBPETokenizer()
    tokenizer.train(text, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(f"Tokenizer saved: {args.output} | vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
