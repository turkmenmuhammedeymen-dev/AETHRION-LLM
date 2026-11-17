"""A compact byte-level BPE tokenizer; all merges are learned from the supplied corpus."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from aethrion.prepare_data import validate_training_approval

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
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
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
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if left < 0 or right < 0 or left >= len(vocab) or right >= len(vocab):
                raise ValueError("Tokenizer merge references an unknown token")
            vocab.append(vocab[left] + vocab[right])
        if any(not isinstance(token_id, int) or token_id < 0 or token_id >= len(vocab) for token_id in ids):
            raise ValueError("Token id is outside this tokenizer vocabulary")
        raw = b"".join(vocab[token_id] for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "aethrion-byte-bpe-v1", "merges": self.merges},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != "aethrion-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        merges = payload.get("merges")
        if not isinstance(merges, list):
            raise ValueError("Tokenizer merges must be a list")
        normalized = []
        for pair in merges:
            if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
                raise ValueError("Each tokenizer merge must contain exactly two integer token IDs")
            normalized.append(tuple(pair))
        tokenizer = cls(normalized)
        # Validate the complete merge graph at load time.
        tokenizer.decode([])
        return tokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AETHRION byte-level BPE tokenizer")
    parser.add_argument("--train", required=True, help="UTF-8 approved train-only split file")
    parser.add_argument("--corpus", required=True, help="Combined corpus path bound to the audit report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--dataset-manifest", required=True,
                        help="Dataset manifest; training_allowed must be explicitly true.")
    parser.add_argument("--source-manifest", required=True,
                        help="Manifest listing every approved training source.")
    parser.add_argument("--source-root", required=True,
                        help="Directory containing the files listed in the source manifest.")
    parser.add_argument("--corpus-report", required=True,
                        help="Audit report produced by prepare_data for the exact corpus.")
    parser.add_argument("--source-review", default=None,
                        help="Evidence-backed per-domain approval CSV for URL-attributed records.")
    args = parser.parse_args()
    verified_splits = validate_training_approval(
        Path(args.dataset_manifest), Path(args.source_manifest), Path(args.source_root),
        Path(args.corpus), Path(args.corpus_report),
        Path(args.source_review) if args.source_review else None,
    )
    train_path = Path(args.train).resolve()
    if train_path != verified_splits["train"]["path"]:
        raise ValueError("Tokenizer training input must be the approved train-only split; validation/test text is forbidden.")
    text = train_path.read_text(encoding="utf-8")
    tokenizer = ByteBPETokenizer()
    tokenizer.train(text, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(f"Tokenizer saved: {args.output} | vocab_size={tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
