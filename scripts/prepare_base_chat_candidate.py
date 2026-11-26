#!/usr/bin/env python3
"""Build a fail-closed BASE tokenizer and CHAT-instruction preparation candidate.

This is preprocessing only. It never changes the official corpus manifest,
approval flags, checkpoints, or model release state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aethrion.specialist_prepare import prepare_text_dataset
from aethrion.tokenizer import ByteBPETokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(base_dir: Path, chat_source: Path, *, vocab_size: int = 2048, max_length: int = 512) -> dict:
    base_dir = base_dir.resolve()
    chat_source = chat_source.resolve()
    manifest_path = base_dir / "manifest.json"
    train_path = base_dir / "train.txt"
    if not manifest_path.is_file() or not train_path.is_file() or not chat_source.is_file():
        raise ValueError("BASE candidate manifest/train split or CHAT source file is missing.")

    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("training_allowed", "owner_approval", "privacy_reviewed"):
        if base_manifest.get(key) is not False:
            raise ValueError(f"BASE candidate must remain fail-closed: {key} must be false.")
    if base_manifest.get("status") != "candidate_only_pending_owner_and_quality_review":
        raise ValueError("Unexpected BASE candidate status; refusing tokenizer preparation.")

    split_hashes = {}
    for split in ("train", "validation", "test"):
        split_info = base_manifest.get("splits", {}).get(split, {})
        split_path = base_dir / split_info.get("file", f"{split}.txt")
        if not split_path.is_file() or sha256(split_path) != split_info.get("sha256"):
            raise ValueError(f"BASE {split} split is missing or its SHA-256 does not match the manifest.")
        split_hashes[split] = sha256(split_path)

    train_text = train_path.read_text(encoding="utf-8")
    if not train_text.strip():
        raise ValueError("BASE train split is empty.")

    tokenizer = ByteBPETokenizer()
    tokenizer.train(train_text, vocab_size=vocab_size)
    tokenizer_path = base_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    corpus_path = base_dir / "corpus.txt"
    corpus_path.write_text(train_text, encoding="utf-8")

    provenance = {
        "format": "aethrion-base-tokenizer-candidate-v1",
        "status": "candidate_only_pending_owner_and_quality_review",
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "tokenizer_path": "tokenizer.json",
        "tokenizer_sha256": sha256(tokenizer_path),
        "corpus_path": "corpus.txt",
        "corpus_sha256": sha256(corpus_path),
        "base_manifest_sha256": sha256(manifest_path),
        "split_sha256": split_hashes,
        "vocab_size": tokenizer.vocab_size,
        "source_file_count": base_manifest.get("source_file_count"),
        "candidate_record_count": base_manifest.get("candidate_record_count"),
        "limitations": [
            "Tokenizer was learned only from the candidate BASE train split; validation and test text were not used.",
            "This is not the official approved corpus/tokenizer and must not be used for model training.",
            "Rights, privacy, provenance, factual quality and content suitability remain pending human review."
        ]
    }
    provenance_path = base_dir / "tokenizer_candidate_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    chat_result = prepare_text_dataset(
        chat_source,
        model_id="chat",
        output_root=base_dir / "chat_candidate",
        license_name="Owner-claimed; third-party rights not independently verified",
        permitted_use="pending human review; training not approved",
        evidence_uri="https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/data/authored/chat/GOVERNANCE.json",
        source_uri="https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/data/authored/chat/records.jsonl",
        seed=1337,
        vocab_size=vocab_size,
        max_length=max_length,
        tokenizer_source_path=tokenizer_path,
    )
    chat_manifest_path = base_dir / "chat_candidate" / "manifests" / "chat.json"
    chat_manifest = json.loads(chat_manifest_path.read_text(encoding="utf-8"))
    if any(chat_manifest.get(key) is not False for key in ("training_allowed", "owner_approval", "privacy_reviewed")):
        raise ValueError("CHAT candidate unexpectedly enabled an approval flag.")
    summary = {
        "status": "candidate_only_pending_owner_and_quality_review",
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "base_tokenizer_sha256": sha256(tokenizer_path),
        "base_corpus_sha256": sha256(corpus_path),
        "chat_source_sha256": sha256(chat_source),
        "chat_dataset_sha256": chat_manifest["dataset_sha256"],
        "chat_tokenizer_sha256": chat_manifest["tokenizer_sha256"],
        "chat_records": chat_result["records"],
        "chat_split_counts": chat_result["split_counts"],
        "chat_manifest": str(chat_manifest_path.relative_to(base_dir)),
    }
    summary_path = base_dir / "chat_candidate" / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/processed/base_candidate"))
    parser.add_argument("--chat-source", type=Path, default=Path("data/authored/chat/records.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    build(args.base_dir, args.chat_source, vocab_size=args.vocab_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
