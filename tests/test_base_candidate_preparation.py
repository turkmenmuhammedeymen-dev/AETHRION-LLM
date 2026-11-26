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
    assert manifest["candidate_record_count"] == 5
    assert manifest["excluded_record_count"] == 0
    review_rows = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(review_rows) == 5
    assert manifest["review_records_file"]["records"] == 5
    assert all(row["split"] == manifest["group_assignment"][row["source_group"]] for row in review_rows)
    assert manifest["group_assignment"]["a.jsonl"] == manifest["group_assignment"]["b.jsonl"]
    assert {manifest["group_assignment"][name] for name in ("a.jsonl", "c.jsonl", "d.jsonl")} == {
        "train", "validation", "test"
    }
    for split in ("train", "validation", "test"):
        path = output / f"{split}.txt"
        assert path.is_file() and path.read_text(encoding="utf-8").strip()
        assert manifest["splits"][split]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()



def test_repeated_chat_answers_are_kept_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "base"
    output = tmp_path / "candidate"
    source.mkdir()
    for index in range(1, 5):
        _write_jsonl(
            source / f"diyalog-{index:02d}.jsonl",
            [{"text": f"Kullanıcı: farklı soru {index}?\\nAETHRION: Aynı yanıt."}],
        )
    for name in ("other-a.jsonl", "other-b.jsonl", "other-c.jsonl"):
        _write_jsonl(source / name, [{"text": f"Bağımsız içerik {name}."}])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    dialogue_rows = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["source_file"].startswith("diyalog-")
    ]
    dialogue_splits = {row["split"] for row in dialogue_rows}
    assert len(dialogue_splits) == 1
    assert len({row["source_group"] for row in dialogue_rows}) == 1
    rows = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dialogue_rows = [row for row in rows if row["source_file"].startswith("diyalog-")]
    assert len(dialogue_rows) == 4
    assert manifest["candidate_record_count"] == 6
    assert manifest["excluded_record_count"] == 1
    assert all("assistant_response_repeated_more_than_three_times" in row["quality_flags"] for row in dialogue_rows)



def test_candidate_preparation_blocks_private_data_patterns(tmp_path: Path) -> None:
    source = tmp_path / "base"
    output = tmp_path / "candidate"
    source.mkdir()
    _write_jsonl(source / "a.jsonl", [{"text": "A safe factual sentence."}])
    _write_jsonl(source / "b.jsonl", [{"text": "Another safe factual sentence."}])
    _write_jsonl(source / "c.jsonl", [{"text": "A third safe factual sentence."}])
    _write_jsonl(source / "d.jsonl", [{"text": "Contact test.user@example.invalid for details."}])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "potential email_like detected" in result.stderr
    assert not (output / "manifest.json").exists()


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
