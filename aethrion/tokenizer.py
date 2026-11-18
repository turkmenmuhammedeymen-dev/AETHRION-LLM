"""A compact byte-level BPE tokenizer; all merges are learned from the supplied corpus."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from aethrion.prepare_data import sha256_file, validate_training_approval

MAX_VOCAB_SIZE = 262144
MAX_TOKEN_BYTES = 1024 * 1024
MAX_TOTAL_VOCAB_BYTES = 64 * 1024 * 1024
MAX_TOKENIZER_FILE_BYTES = 16 * 1024 * 1024


def _validate_merge_graph(merges: list[tuple[int, int]]) -> None:
    """Validate merge references and bound expanded vocabulary memory without materializing tokens."""
    if len(merges) > MAX_VOCAB_SIZE - 256:
        raise ValueError("Tokenizer merge count exceeds the supported vocabulary limit.")
    lengths = [1] * 256
    seen: set[tuple[int, int]] = set()
    total_bytes = 256
    for index, pair in enumerate(merges):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(type(value) is not int for value in pair):
            raise ValueError("Each tokenizer merge must contain exactly two integer token IDs")
        left, right = pair
        current_vocab_size = 256 + index
        if left < 0 or right < 0 or left >= current_vocab_size or right >= current_vocab_size:
            raise ValueError("Tokenizer merge references an unknown or forward token")
        normalized = (left, right)
        if normalized in seen:
            raise ValueError("Tokenizer merge graph contains a duplicate pair")
        seen.add(normalized)
        token_bytes = lengths[left] + lengths[right]
        if token_bytes > MAX_TOKEN_BYTES:
            raise ValueError("Tokenizer merge expands a token beyond the supported byte limit")
        total_bytes += token_bytes
        if total_bytes > MAX_TOTAL_VOCAB_BYTES:
            raise ValueError("Tokenizer expanded vocabulary exceeds the supported memory limit")
        lengths.append(token_bytes)

class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self._rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @staticmethod
    def _bytes(text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def train(self, text: str, vocab_size: int = 2048, min_frequency: int = 2) -> None:
        if type(vocab_size) is not int or not 256 <= vocab_size <= MAX_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be an integer between 256 and {MAX_VOCAB_SIZE}")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")
        sequences = [self._bytes(line) for line in text.splitlines() if line.strip()]
        merges: list[tuple[int, int]] = []
        token_lengths = [1] * 256
        total_vocab_bytes = 256
        while 256 + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < min_frequency:
                break
            token_bytes = token_lengths[pair[0]] + token_lengths[pair[1]]
            if (
                token_bytes > MAX_TOKEN_BYTES
                or total_vocab_bytes + token_bytes > MAX_TOTAL_VOCAB_BYTES
            ):
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            token_lengths.append(token_bytes)
            total_vocab_bytes += token_bytes
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
        _validate_merge_graph(self.merges)
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if left < 0 or right < 0 or left >= len(vocab) or right >= len(vocab):
                raise ValueError("Tokenizer merge references an unknown token")
            vocab.append(vocab[left] + vocab[right])
        if any(type(token_id) is not int or token_id < 0 or token_id >= len(vocab) for token_id in ids):
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
        source = Path(path)
        if not source.is_file() or source.stat().st_size > MAX_TOKENIZER_FILE_BYTES:
            raise ValueError("Tokenizer file is missing or exceeds the supported size limit")
        payload = json.loads(source.read_text(encoding="utf-8"))
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
        _validate_merge_graph(normalized)
        return cls(normalized)

def validate_tokenizer_provenance(
    tokenizer_path: str | Path,
    corpus_path: str | Path,
    dataset_manifest_path: str | Path,
    source_manifest_path: str | Path,
    source_root: str | Path,
    corpus_report_path: str | Path,
    source_review_path: str | Path | None = None,
) -> tuple[dict[str, dict], str]:
    """Verify that the tokenizer was trained from the current approved train-only partition."""
    tokenizer_file = Path(tokenizer_path).resolve()
    corpus_file = Path(corpus_path).resolve()
    dataset_file = Path(dataset_manifest_path).resolve()
    source_file = Path(source_manifest_path).resolve()
    report_file = Path(corpus_report_path).resolve()
    review_file = Path(source_review_path).resolve() if source_review_path is not None else None
    verified_splits = validate_training_approval(
        dataset_file, source_file, Path(source_root), corpus_file, report_file, review_file
    )
    if not tokenizer_file.is_file():
        raise ValueError(f"Training blocked: tokenizer is missing: {tokenizer_file}")
    # Loading also validates the tokenizer's actual merge graph before trusting its hash.
    ByteBPETokenizer.load(tokenizer_file)
    provenance_path = Path(str(tokenizer_file) + ".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Training blocked: tokenizer provenance sidecar is missing.")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Training blocked: tokenizer provenance sidecar is invalid.") from exc
    expected = {
        "format": "aethrion-tokenizer-provenance-v1",
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "corpus_sha256": sha256_file(corpus_file),
        "corpus_report_sha256": sha256_file(report_file),
        "dataset_manifest_sha256": sha256_file(dataset_file),
        "source_manifest_sha256": sha256_file(source_file),
        "source_review_sha256": (
            sha256_file(review_file) if review_file is not None else None
        ),
        "train_split_sha256": verified_splits["train"]["sha256"],
        "validation_split_sha256": verified_splits["validation"]["sha256"],
        "test_split_sha256": verified_splits["test"]["sha256"],
    }
    if provenance != expected:
        raise ValueError(
            "Training blocked: tokenizer provenance does not match the approved corpus, "
            "source manifests and split hashes."
        )
    return verified_splits, sha256_file(provenance_path)


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
    tokenizer_path = Path(args.output)
    tokenizer.save(tokenizer_path)
    source_review_path = Path(args.source_review) if args.source_review else None
    provenance = {
        "format": "aethrion-tokenizer-provenance-v1",
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "corpus_sha256": sha256_file(Path(args.corpus)),
        "corpus_report_sha256": sha256_file(Path(args.corpus_report)),
        "dataset_manifest_sha256": sha256_file(Path(args.dataset_manifest)),
        "source_manifest_sha256": sha256_file(Path(args.source_manifest)),
        "source_review_sha256": (
            sha256_file(source_review_path) if source_review_path is not None else None
        ),
        "train_split_sha256": verified_splits["train"]["sha256"],
        "validation_split_sha256": verified_splits["validation"]["sha256"],
        "test_split_sha256": verified_splits["test"]["sha256"],
    }
    provenance_path = Path(str(tokenizer_path) + ".provenance.json")
    temporary_path = Path(str(provenance_path) + ".tmp")
    temporary_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(provenance_path)
    print(f"Tokenizer saved: {tokenizer_path} | vocab_size={tokenizer.vocab_size}")
    print(f"Tokenizer provenance: {provenance_path}")

if __name__ == "__main__":
    main()
