#!/usr/bin/env python3
"""Build a non-training BASE corpus candidate with deterministic group splits.

This tool never changes approval manifests and never enables training. It reads
JSONL source files from data/authored/base and writes a separate candidate
snapshot under data/processed/base_candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build_candidate(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise ValueError(f"BASE source directory does not exist: {source_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Candidate output must not be inside the authored source directory.")

    files = sorted(source_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL source files found in {source_dir}")

    file_hashes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    group_names = [path.name for path in files]
    union = UnionFind(group_names)
    seen: dict[str, dict[str, Any]] = {}
    duplicate_count = 0

    for path in files:
        raw = path.read_bytes()
        file_hashes[path.name] = {
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        }
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path.name}: not valid UTF-8") from exc

        for line_number, line in enumerate(decoded.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: each JSONL row must be an object")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path.name}:{line_number}: missing or empty text")
            key = normalized_text(text)
            if not key:
                raise ValueError(f"{path.name}:{line_number}: text is empty after normalization")
            group = path.name
            if key in seen:
                union.union(group, seen[key]["source_group"])
                duplicate_count += 1
                continue
            item = {
                "text": text.strip(),
                "source_file": path.name,
                "source_line": line_number,
                "source_group": group,
                "text_sha256": sha256_bytes(key.encode("utf-8")),
            }
            seen[key] = item
            rows.append(item)

    if len(rows) < 3:
        raise ValueError("At least three unique text records are required for candidate splits.")

    components: dict[str, list[str]] = defaultdict(list)
    for group in group_names:
        components[union.find(group)].append(group)
    ordered_components = sorted(
        components,
        key=lambda root: hashlib.sha256("\n".join(sorted(components[root])).encode("utf-8")).hexdigest(),
    )
    count = len(ordered_components)
    if count < 3:
        raise ValueError("At least three independent source groups are required after duplicate grouping.")

    train_count = max(1, round(count * 0.8))
    validation_count = max(1, round(count * 0.1))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    assignment: dict[str, str] = {}
    for index, root in enumerate(ordered_components):
        split = (
            "train" if index < train_count
            else "validation" if index < train_count + validation_count
            else "test"
        )
        for member in components[root]:
            assignment[member] = split

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    for item in rows:
        split_rows[assignment[item["source_group"]]].append(item)

    if any(not split_rows[name] for name in split_rows):
        raise ValueError("A candidate split is empty; refusing to write an incomplete snapshot.")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    for split, items in split_rows.items():
        items.sort(key=lambda item: (item["source_file"], item["source_line"]))
        content = "\n\n".join(item["text"] for item in items) + "\n"
        filename = f"{split}.txt"
        atomic_write(output_dir / filename, content)
        split_files[split] = {
            "file": filename,
            "records": len(items),
            "groups": len({item["source_group"] for item in items}),
            "sha256": sha256_bytes(content.encode("utf-8")),
        }

    review_rows = []
    for split, items in split_rows.items():
        for item in items:
            review_rows.append({**item, "split": split})
    review_jsonl = "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in review_rows
    )
    atomic_write(output_dir / "records.jsonl", review_jsonl)

    manifest = {
        "schema_version": 1,
        "project": "AETHRION-LLM",
        "status": "candidate_only_pending_owner_and_quality_review",
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "source_directory": "data/authored/base",
        "source_files": file_hashes,
        "source_file_count": len(files),
        "input_record_count": sum(
            1 for path in files for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
        "unique_record_count": len(rows),
        "duplicate_records_removed": duplicate_count,
        "independent_group_count_after_duplicate_union": count,
        "split_policy": "deterministic 80/10/10 by source-file group; exact-normalized duplicates union source groups",
        "normalization_for_duplicate_detection_only": "Unicode NFKC, casefold, collapse whitespace; stored text is otherwise preserved",
        "group_assignment": assignment,
        "splits": split_files,
        "review_records_file": {"file": "records.jsonl", "records": len(review_rows), "sha256": sha256_bytes(review_jsonl.encode("utf-8"))},
        "record_audit": [
            {
                "source_file": item["source_file"],
                "source_line": item["source_line"],
                "source_group": item["source_group"],
                "split": assignment[item["source_group"]],
                "normalized_text_sha256": item["text_sha256"],
            }
            for item in rows
        ],
        "limitations": [
            "This is a structural candidate preparation, not a rights, privacy, factual-quality, or safety approval.",
            "Source groups are approximated by source filename because current BASE rows do not carry stable group IDs.",
            "Exact normalized duplicate texts are removed and their source groups are kept in one split.",
            "No tokenizer is trained, no checkpoint is created, and no training gate is opened.",
        ],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    atomic_write(output_dir / "manifest.json", manifest_text)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fail-closed BASE candidate corpus.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/authored/base"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/base_candidate"))
    args = parser.parse_args()
    manifest = build_candidate(args.source_dir, args.output_dir)
    print(json.dumps({
        "status": manifest["status"],
        "training_allowed": manifest["training_allowed"],
        "source_files": manifest["source_file_count"],
        "records": manifest["unique_record_count"],
        "duplicates_removed": manifest["duplicate_records_removed"],
        "groups": manifest["independent_group_count_after_duplicate_union"],
        "splits": manifest["splits"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
