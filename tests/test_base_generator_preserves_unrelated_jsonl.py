from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_base_generator_preserves_unrelated_jsonl(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_dir = repo / "data" / "authored" / "base"
    chat_dir = repo / "data" / "authored" / "chat"
    base_dir.mkdir(parents=True)
    chat_dir.mkdir(parents=True)

    source_script = Path("data/authored/base/make_base.py").read_text(encoding="utf-8")
    (base_dir / "make_base.py").write_text(source_script, encoding="utf-8")
    (repo / "data" / "authored" / "make_expand.py").write_text(
        "def extra_groups():\n    return {}\n", encoding="utf-8"
    )
    (chat_dir / "records.jsonl").write_text("", encoding="utf-8")

    protected = base_dir / "manually-reviewed.jsonl"
    protected_record = {"id": "manual-keep-001", "text": "Reviewed record to preserve."}
    protected.write_text(json.dumps(protected_record, ensure_ascii=False) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(base_dir / "make_base.py")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert protected.exists(), "The generator deleted an unrelated JSONL file."
    assert json.loads(protected.read_text(encoding="utf-8")) == protected_record
    assert list(base_dir.glob("bilim-*.jsonl")), "The generator did not write its own outputs."
