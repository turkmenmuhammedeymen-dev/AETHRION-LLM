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
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
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

    # Identical assistant responses across CHAT-derived dialogue files must
    # remain in one split; otherwise a held-out split can contain a memorized
    # answer seen in training under a different prompt.
    response_first_group: dict[str, str] = {}
    response_counts: Counter[str] = Counter()
    for item in rows:
        if item["source_file"].startswith("diyalog-") and "AETHRION:" in item["text"]:
            response = item["text"].split("AETHRION:", 1)[1].strip()
            response_key = normalized_text(response)
            if response_key:
                response_counts[response_key] += 1
                if response_key in response_first_group:
                    union.union(item["source_group"], response_first_group[response_key])
                else:
                    response_first_group[response_key] = item["source_group"]

    response_seen: Counter[str] = Counter()
    for item in rows:
        flags: list[str] = []
        exclusion_reasons: list[str] = []
        if re.search(r"için en bilinen özellik", item["text"], re.IGNORECASE):
            flags.append("generic_animal_property_template")
            exclusion_reasons.append("generic_animal_property_template")
        if re.search(r"\\b(yılın|ayın)\\s+\\w+\\.\\s+ayıdır", item["text"], re.IGNORECASE):
            flags.append("malformed_ordinal_month_phrase")
            exclusion_reasons.append("malformed_ordinal_month_phrase")
        if item["source_file"].startswith("diyalog-") and "AETHRION:" in item["text"]:
            response_key = normalized_text(item["text"].split("AETHRION:", 1)[1].strip())
            if response_key:
                item["assistant_response_sha256"] = sha256_bytes(response_key.encode("utf-8"))
                if response_counts[response_key] > 3:
                    flags.append("assistant_response_repeated_more_than_three_times")
                response_seen[response_key] += 1
                if response_seen[response_key] > 3:
                    flags.append("assistant_response_excess_duplicate")
                    exclusion_reasons.append("assistant_response_cap_3")
            else:
                item["assistant_response_sha256"] = None
        item["quality_flags"] = flags
        item["candidate_included"] = not exclusion_reasons
        item["candidate_exclusion_reasons"] = exclusion_reasons

    eligible_rows = [item for item in rows if item["candidate_included"]]
    if len(eligible_rows) < 3:
        raise ValueError("At least three eligible text records are required for candidate splits.")

    all_components: dict[str, list[str]] = defaultdict(list)
    for group in group_names:
        all_components[union.find(group)].append(group)
    eligible_roots = {union.find(item["source_group"]) for item in eligible_rows}
    components = {root: all_components[root] for root in eligible_roots}
    ordered_components = sorted(
        components,
        key=lambda root: hashlib.sha256("\\n".join(sorted(components[root])).encode("utf-8")).hexdigest(),
    )
    count = len(ordered_components)
    if count < 3:
        raise ValueError("At least three independent eligible source groups are required after duplicate grouping.")

    train_count = max(1, round(count * 0.8))
    validation_count = max(1, round(count * 0.1))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    assigned_groups: dict[str, str] = {}
    for index, root in enumerate(ordered_components):
        split = (
            "train" if index < train_count
            else "validation" if index < train_count + validation_count
            else "test"
        )
        for member in components[root]:
            assigned_groups[member] = split
    assignment: dict[str, str | None] = {
        group: assigned_groups.get(group) for group in group_names
    }

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    for item in eligible_rows:
        split = assignment[item["source_group"]]
        if split is None:
            raise ValueError(f"No split assignment for eligible source group {item['source_group']}")
        split_rows[split].append(item)

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

    review_rows = [
        {**item, "split": assignment.get(item["source_group"])}
        for item in rows
    ]
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
        "candidate_record_count": len(eligible_rows),
        "excluded_record_count": len(rows) - len(eligible_rows),
        "excluded_by_reason": dict(Counter(reason for item in rows for reason in item["candidate_exclusion_reasons"])),
        "duplicate_records_removed": duplicate_count,
        "independent_group_count_after_duplicate_union": len(all_components),
        "candidate_group_component_count": count,
        "split_policy": "deterministic 80/10/10 by source-file group; exact-normalized record duplicates and repeated CHAT assistant responses union source groups; repeated responses capped at three candidate rows",
        "normalization_for_duplicate_detection_only": "Unicode NFKC, casefold, collapse whitespace; stored text is otherwise preserved",
        "group_assignment": assignment,
        "quality_flag_counts": dict(Counter(flag for item in rows for flag in item["quality_flags"])),
        "quality_review_required_records": sum(bool(item["quality_flags"]) for item in rows),
        "candidate_exclusion_policy": "exclude malformed ordinal-month and generic animal-property templates; cap identical CHAT assistant responses at three rows",
        "splits": split_files,
        "review_records_file": {"file": "records.jsonl", "records": len(review_rows), "sha256": sha256_bytes(review_jsonl.encode("utf-8"))},
        "record_audit": [
            {
                "source_file": item["source_file"],
                "source_line": item["source_line"],
                "source_group": item["source_group"],
                "split": assignment.get(item["source_group"]),
                "candidate_included": item["candidate_included"],
                "candidate_exclusion_reasons": item["candidate_exclusion_reasons"],
                "normalized_text_sha256": item["text_sha256"],
                "quality_flags": item["quality_flags"],
            }
            for item in rows
        ],
        "limitations": [
            "This is a structural candidate preparation, not a rights, privacy, factual-quality, or safety approval.",
            "Source groups are approximated by source filename because current BASE rows do not carry stable group IDs.",
            "Exact normalized duplicate texts are removed and their source groups are kept in one split.",
            "Repeated assistant responses across CHAT-derived dialogue files are grouped into one split to reduce answer leakage and capped at three rows per exact normalized response.",
            "Clearly malformed ordinal-month and generic animal-property templates are excluded from train/validation/test but retained in records.jsonl with exclusion reasons for audit.",
            "Heuristic quality flags are review aids only; they do not automatically certify or repair content.",
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
