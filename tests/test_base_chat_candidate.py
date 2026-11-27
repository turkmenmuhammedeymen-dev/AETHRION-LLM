from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path("scripts/prepare_base_chat_candidate.py").resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(group_id: str) -> str:
    value = int(hashlib.sha256(f"1337:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def test_base_tokenizer_and_chat_candidate_remain_fail_closed(tmp_path: Path) -> None:
    base = tmp_path / "base_candidate"
    base.mkdir()
    split_text = {
        "train": "Türkçe temel bilgi. Yapay zekâ metinleri işler.\n",
        "validation": "Doğrulama metni.\n",
        "test": "Test metni.\n",
    }
    split_manifest = {}
    for split, content in split_text.items():
        path = base / f"{split}.txt"
        path.write_text(content, encoding="utf-8")
        split_manifest[split] = {
            "file": path.name,
            "records": len(content.splitlines()),
            "groups": 1,
            "sha256": _sha(path),
        }
    (base / "manifest.json").write_text(
        json.dumps({
            "status": "candidate_only_pending_owner_and_quality_review",
            "training_allowed": False,
            "owner_approval": False,
            "privacy_reviewed": False,
            "source_file_count": 3,
            "candidate_record_count": 3,
            "splits": split_manifest,
        }),
        encoding="utf-8",
    )

    chat_rows = []
    seen_splits = set()
    index = 0
    while len(seen_splits) < 3:
        group_id = f"chat-group-{index}"
        split = _split(group_id)
        if split not in seen_splits:
            chat_rows.append({
                "id": f"chat-{index}",
                "group_id": group_id,
                "prompt": f"Bu örnek soru {index} nedir?",
                "response": f"Bu örnek yanıt {index}.",
            })
            seen_splits.add(split)
        index += 1
    chat_source = tmp_path / "chat.jsonl"
    chat_source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chat_rows),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--base-dir", str(base),
            "--chat-source", str(chat_source),
            "--vocab-size", "512",
            "--max-length", "128",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    tokenizer = base / "tokenizer.json"
    corpus = base / "corpus.txt"
    provenance = json.loads((base / "tokenizer_candidate_provenance.json").read_text(encoding="utf-8"))
    assert tokenizer.is_file() and corpus.read_text(encoding="utf-8") == split_text["train"]
    assert provenance["training_allowed"] is False
    assert provenance["owner_approval"] is False
    assert provenance["privacy_reviewed"] is False
    assert provenance["tokenizer_sha256"] == _sha(tokenizer)
    assert provenance["corpus_sha256"] == _sha(corpus)

    chat_root = base / "chat_candidate" / "manifests"
    chat_manifest = json.loads((chat_root / "chat.json").read_text(encoding="utf-8"))
    assert chat_manifest["training_allowed"] is False
    assert chat_manifest["owner_approval"] is False
    assert chat_manifest["privacy_reviewed"] is False
    assert chat_manifest["tokenizer_sha256"] == _sha(tokenizer)
    assert chat_manifest["preparation"]["split_counts"].keys() == {"train", "validation", "test"}

    arrays = np.load(chat_root / "chat" / "dataset.npz", allow_pickle=False)
    group_splits = {}
    for group_id, split in zip(arrays["group_id"].tolist(), arrays["split"].tolist()):
        group_splits.setdefault(group_id, set()).add(split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert set(arrays["split"].tolist()) == {"train", "validation", "test"}
