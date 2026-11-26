from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/prepare_base_candidate.py").resolve()


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_candidate_preparation_deduplicates_and_keeps_groups_together(tmp_path: Path) -> None:
    source = tmp_path / "base"
    output = tmp_path / "candidate"
    source.mkdir()
    _write_jsonl(source / "a.jsonl", [{"text": "Alpha fact."}, {"text": "Shared fact."}])
    _write_jsonl(source / "b.jsonl", [{"text": "Beta fact."}, {"text": " shared   FACT. "}])
    _write_jsonl(source / "c.jsonl", [{"text": "Gamma fact."}])
    _write_jsonl(source / "d.jsonl", [{"text": "Delta fact."}])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_allowed"] is False
    assert manifest["owner_approval"] is False
    assert manifest["privacy_reviewed"] is False
    assert manifest["duplicate_records_removed"] == 1
    assert manifest["unique_record_count"] == 5
    assert manifest["group_assignment"]["a.jsonl"] == manifest["group_assignment"]["b.jsonl"]
    assert {manifest["group_assignment"][name] for name in ("a.jsonl", "c.jsonl", "d.jsonl")} == {
        "train", "validation", "test"
    }
    for split in ("train", "validation", "test"):
        path = output / f"{split}.txt"
        assert path.is_file() and path.read_text(encoding="utf-8").strip()
        assert manifest["splits"][split]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_preparation_rejects_invalid_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "base"
    output = tmp_path / "candidate"
    source.mkdir()
    (source / "broken.jsonl").write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid JSON" in result.stderr
    assert not (output / "manifest.json").exists()
