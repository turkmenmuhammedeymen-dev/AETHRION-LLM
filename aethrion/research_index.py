"""Build a provenance-preserving offline research index from approved local documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from aethrion.tokenizer import ByteBPETokenizer

APPROVAL_FORMAT = "aethrion-research-index-approval-v1"
INDEX_FORMAT = "aethrion-local-research-index-v1"


class ResearchIndexError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or chr(92) in relative or "://" in relative:
        raise ResearchIndexError("Only relative local research source paths are accepted.")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ResearchIndexError("Research source path contains unsafe components.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ResearchIndexError("Research source path escapes the approved data root.") from exc
    return resolved


def build_local_research_index(
    records_path: str | Path,
    approval_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index local UTF-8 JSONL text only after each source's index-use approval."""
    if not 16 <= chunk_tokens <= 2048 or not 0 <= overlap_tokens < chunk_tokens:
        raise ResearchIndexError("Invalid chunk_tokens or overlap_tokens.")
    records_file = Path(records_path).resolve()
    approval_file = Path(approval_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    try:
        approval = json.loads(approval_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResearchIndexError("Research index approval manifest is missing or invalid.") from exc
    if not isinstance(approval, dict) or approval.get("format") != APPROVAL_FORMAT:
        raise ResearchIndexError("Unsupported research index approval format.")
    for key in ("index_allowed", "owner_approval", "privacy_reviewed"):
        if type(approval.get(key)) is not bool or approval[key] is not True:
            raise ResearchIndexError(f"Research indexing blocked: {key} must be explicitly true.")
    if not records_file.is_file() or not tokenizer_file.is_file():
        raise ResearchIndexError("Records JSONL and local tokenizer files are required.")
    sources = approval.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ResearchIndexError("At least one approved local research source is required.")
    source_by_id = {}
    data_root = approval_file.parent.resolve()
    for source in sources:
        if not isinstance(source, dict):
            raise ResearchIndexError("Each research source record must be an object.")
        required = ("source_id", "source_path", "source_uri", "license",
                    "permitted_use", "evidence_uri", "sha256")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in required):
            raise ResearchIndexError("Research source is missing provenance/license evidence fields.")
        if source.get("allow_index") is not True or source.get("privacy_reviewed") is not True:
            raise ResearchIndexError("Research indexing blocked: source index permission/privacy review missing.")
        local = _safe_path(data_root, source["source_path"])
        if not local.is_file() or _sha256(local) != source["sha256"].lower():
            raise ResearchIndexError("Research source file missing or SHA-256 mismatch.")
        if source["source_id"] in source_by_id:
            raise ResearchIndexError("Duplicate research source_id.")
        source_by_id[source["source_id"]] = source

    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    stride = chunk_tokens - overlap_tokens
    chunks: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = records_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResearchIndexError("Research records JSONL is unreadable or not UTF-8.") from exc
    if not lines or len(lines) > 100_000:
        raise ResearchIndexError("Research records must contain 1–100,000 lines.")
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResearchIndexError(f"Invalid JSONL at line {line_number}.") from exc
        if not isinstance(record, dict):
            raise ResearchIndexError(f"Research record at line {line_number} must be an object.")
        doc_id, source_id, title, text = (
            record.get("doc_id"), record.get("source_id"),
            record.get("title"), record.get("text")
        )
        if any(not isinstance(value, str) or not value.strip() for value in (doc_id, source_id, title, text)):
            raise ResearchIndexError(f"Research record {line_number} lacks doc_id/source_id/title/text.")
        if doc_id in seen_ids:
            raise ResearchIndexError(f"Duplicate research doc_id: {doc_id}.")
        seen_ids.add(doc_id)
        source = source_by_id.get(source_id)
        if source is None:
            raise ResearchIndexError(f"Research record {doc_id} refers to an unapproved source.")
        if len(text.encode("utf-8")) > 1_000_000:
            raise ResearchIndexError(f"Research record {doc_id} exceeds the 1 MiB text limit.")
        tokens = tokenizer.encode(text)
        if not tokens:
            continue
        for start in range(0, len(tokens), stride):
            part = tokens[start:start + chunk_tokens]
            if not part:
                continue
            chunks.append(part)
            excerpt = tokenizer.decode(part)[:1200]
            metadata.append({
                "chunk_id": len(metadata),
                "doc_id": doc_id,
                "title": title[:500],
                "source_uri": source["source_uri"],
                "license": source["license"],
                "evidence_uri": source["evidence_uri"],
                "source_sha256": source["sha256"].lower(),
                "excerpt": excerpt,
            })
            if start + chunk_tokens >= len(tokens):
                break
    if not chunks:
        raise ResearchIndexError("No indexable text chunks were produced.")
    max_length = max(len(chunk) for chunk in chunks)
    token_array = np.zeros((len(chunks), max_length), dtype=np.int64)
    mask_array = np.zeros((len(chunks), max_length), dtype=np.bool_)
    for index, chunk in enumerate(chunks):
        token_array[index, :len(chunk)] = np.asarray(chunk, dtype=np.int64)
        mask_array[index, :len(chunk)] = True
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "passages.npz"
    metadata_path = output / "metadata.json"
    processor_path = output / "processor.json"
    np.savez_compressed(index_path, passage_ids=token_array, passage_mask=mask_array)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    repo_root = Path(project_root or Path.cwd()).resolve()
    try:
        index_relative = index_path.resolve().relative_to(repo_root).as_posix()
        metadata_relative = metadata_path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ResearchIndexError("Research index output must be stored inside the repository root.") from exc
    processor = {
        "format": INDEX_FORMAT,
        "tokenizer_sha256": _sha256(tokenizer_file),
        "index_path": index_relative,
        "index_sha256": _sha256(index_path),
        "metadata_path": metadata_relative,
        "metadata_sha256": _sha256(metadata_path),
        "passage_count": len(chunks),
        "source_count": len(source_by_id),
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "execution_policy": "local_only",
    }
    processor_path.write_text(json.dumps(processor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return processor


def main():
    parser = argparse.ArgumentParser(description="Build an offline AETHRION-RESEARCH index from approved local text.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    result = build_local_research_index(
        args.records, args.approval, args.tokenizer, args.output_dir,
        chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens,
        project_root=args.project_root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
