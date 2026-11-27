#!/usr/bin/env python3
"""Read-only structural audit for authored BASE and CHAT candidate data.

This tool never changes manifests, approvals, training flags, checkpoints, or
release state. Its outputs are review artifacts, not training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_jsonl(path: Path, required_fields: tuple[str, ...], text_field: str):
    rows, errors, empty_text, missing = [], [], [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            errors.append({"line": line_no, "error": "blank_line"})
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_no, "error": "invalid_json", "detail": str(exc)})
            continue
        if not isinstance(obj, dict):
            errors.append({"line": line_no, "error": "record_not_object"})
            continue
        absent = [key for key in required_fields if not isinstance(obj.get(key), str) or not obj[key].strip()]
        if absent:
            missing.append({"line": line_no, "fields": absent})
            continue
        value = obj.get(text_field)
        if isinstance(value, str) and value.strip():
            rows.append((line_no, obj, value))
        else:
            empty_text.append(line_no)
    return rows, errors, empty_text, missing, len(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("base-data-audit"))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    base_dir = root / "data/authored/base"
    chat_path = root / "data/authored/chat/records.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    base_files = sorted(base_dir.glob("*.jsonl"))
    report = {
        "format": "aethrion-base-structural-audit-v1",
        "scope": "all authored BASE JSONL files and the authored CHAT records file",
        "training_authorized": False,
        "approval_flags_changed": False,
        "content_accuracy_or_rights_certified": False,
        "base": {"file_count": len(base_files), "total_bytes": 0, "record_count": 0,
                 "invalid_json_or_blank_lines": 0, "empty_text_records": 0,
                 "missing_text_records": 0, "duplicate_normalized_text_records": 0,
                 "duplicate_ids": 0, "files": []},
        "chat": {"file_present": chat_path.is_file(), "record_count": 0,
                 "invalid_json_or_blank_lines": 0, "missing_required_fields": 0,
                 "duplicate_ids": 0, "duplicate_group_ids": 0,
                 "duplicate_normalized_prompts": 0, "duplicate_prompt_response_pairs": 0},
        "limitations": [
            "Structural checks do not establish factual accuracy, ownership, licensing, privacy, consent, or suitability for training.",
            "BASE grouping is by source filename; this is a candidate grouping boundary, not proof that sources are independent.",
            "No model training, checkpoint modification, approval, or release action is performed."
        ]
    }
    all_text = defaultdict(list)
    text_counts, ids = Counter(), Counter()
    candidate_blocks = []
    for path in base_files:
        raw_bytes = path.stat().st_size
        rows, errors, empty, missing, line_count = audit_jsonl(path, ("text",), "text")
        report["base"]["total_bytes"] += raw_bytes
        report["base"]["record_count"] += line_count
        report["base"]["invalid_json_or_blank_lines"] += len(errors)
        report["base"]["empty_text_records"] += len(empty)
        report["base"]["missing_text_records"] += len(missing)
        for _, obj, value in rows:
            text_counts[norm(value)] += 1
            if obj.get("id") is not None:
                ids[str(obj["id"])] += 1
            all_text[path.name].append(value.strip())
        report["base"]["files"].append({
            "path": path.relative_to(root).as_posix(),
            "bytes": raw_bytes,
            "sha256": sha256(path),
            "valid_text_records": len(rows),
            "jsonl_errors": errors,
            "empty_text_lines": empty,
            "missing_text_fields": missing,
            "candidate_group": path.name
        })
        if rows:
            candidate_blocks.append("\n".join(value.strip() for _, _, value in rows))
    report["base"]["duplicate_normalized_text_records"] = sum(n - 1 for n in text_counts.values() if n > 1)
    report["base"]["duplicate_ids"] = sum(n - 1 for n in ids.values() if n > 1)
    (out / "base-candidate-corpus.txt").write_text("\n\n".join(candidate_blocks) + "\n", encoding="utf-8")

    chat_rows, chat_errors, chat_empty, chat_missing, chat_line_count = audit_jsonl(
        chat_path, ("id", "group_id", "prompt", "response"), "prompt"
    ) if chat_path.is_file() else ([], [], [], [], 0)
    chat_ids, groups, prompts, pairs = Counter(), Counter(), Counter(), Counter()
    valid_chat = []
    for _, obj, prompt in chat_rows:
        chat_ids[obj["id"]] += 1
        groups[obj["group_id"]] += 1
        prompts[norm(prompt)] += 1
        pairs[(norm(prompt), norm(obj["response"]))] += 1
        valid_chat.append(obj)
    report["chat"].update({
        "record_count": chat_line_count,
        "invalid_json_or_blank_lines": len(chat_errors),
        "missing_required_fields": len(chat_missing) + len(chat_empty),
        "duplicate_ids": sum(n - 1 for n in chat_ids.values() if n > 1),
        "duplicate_group_ids": sum(n - 1 for n in groups.values() if n > 1),
        "duplicate_normalized_prompts": sum(n - 1 for n in prompts.values() if n > 1),
        "duplicate_prompt_response_pairs": sum(n - 1 for n in pairs.values() if n > 1),
        "sha256": sha256(chat_path) if chat_path.is_file() else None,
        "jsonl_errors": chat_errors,
        "missing_fields": chat_missing,
        "empty_prompt_lines": chat_empty
    })
    with (out / "chat-instruct-candidate.jsonl").open("w", encoding="utf-8") as stream:
        for obj in valid_chat:
            stream.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_files": report["base"]["file_count"],
        "base_records": report["base"]["record_count"],
        "base_errors": report["base"]["invalid_json_or_blank_lines"],
        "base_empty_text": report["base"]["empty_text_records"],
        "base_missing_text": report["base"]["missing_text_records"],
        "base_duplicate_text_rows": report["base"]["duplicate_normalized_text_records"],
        "chat_records": report["chat"]["record_count"],
        "chat_errors": report["chat"]["invalid_json_or_blank_lines"],
        "training_authorized": False,
        "report": str(out / "report.json")
    }, ensure_ascii=False))
    return 0 if report["base"]["file_count"] > 0 and report["base"]["invalid_json_or_blank_lines"] == 0 and report["base"]["empty_text_records"] == 0 and report["base"]["missing_text_records"] == 0 and report["chat"]["invalid_json_or_blank_lines"] == 0 and report["chat"]["missing_required_fields"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
