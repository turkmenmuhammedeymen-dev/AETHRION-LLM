"""Fault-injection tests for atomic progress/report JSON publication."""

from __future__ import annotations

import json

import pytest

from aethrion import checkpoint


def test_interrupted_json_update_preserves_previous_complete_document(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    original = {"step": 12, "status": "running", "detail": "previous complete state"}
    checkpoint.write_json_atomic(original, target)

    real_replace = checkpoint.os.replace

    def fail_before_publish(source, destination):
        raise OSError("simulated interruption before JSON publish")

    monkeypatch.setattr(checkpoint.os, "replace", fail_before_publish)
    replacement = {"step": 13, "status": "interrupted", "detail": "new state"}
    with pytest.raises(OSError, match="before JSON publish"):
        checkpoint.write_json_atomic(replacement, target)

    monkeypatch.setattr(checkpoint.os, "replace", real_replace)
    assert json.loads(target.read_text(encoding="utf-8")) == original
    assert not (tmp_path / "progress.json.tmp").exists()


def test_interrupted_first_json_write_does_not_publish_partial_document(tmp_path, monkeypatch):
    target = tmp_path / "training_report.json"

    def fail_before_publish(source, destination):
        raise OSError("simulated interruption before first JSON publish")

    monkeypatch.setattr(checkpoint.os, "replace", fail_before_publish)
    with pytest.raises(OSError, match="before first JSON publish"):
        checkpoint.write_json_atomic({"status": "running", "step": 1}, target)

    assert not target.exists()
    assert not (tmp_path / "training_report.json.tmp").exists()
