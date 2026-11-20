"""Fault-injection tests for interrupted checkpoint publication."""

from __future__ import annotations

import pytest
import torch

from aethrion import checkpoint


def _payload(step: int) -> dict:
    return {
        "format_version": 4,
        "step": step,
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {},
        "scaler": None,
        "model_config": {"width": 1},
        "data_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "rng_state": {"python": (1, 2, 3)},
        "training_config": {"seed": 7, "learning_rate": 0.001},
    }


def test_interrupted_publication_recovers_previous_generation(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint_last.pt"
    checkpoint.save_checkpoint(_payload(1), target)

    real_replace = checkpoint.os.replace
    calls = 0

    def fail_before_checksum_publish(source, destination):
        nonlocal calls
        # Existing generation is copied to .previous (two replaces), then
        # the new checkpoint is published (third replace). Simulate a crash
        # before its matching checksum sidecar can be published.
        calls += 1
        if calls == 4:
            raise OSError("simulated process interruption before checksum publish")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_before_checksum_publish)

    with pytest.raises(OSError, match="simulated process interruption"):
        checkpoint.save_checkpoint(_payload(2), target)

    monkeypatch.setattr(checkpoint.os, "replace", real_replace)

    recovered = checkpoint.load_checkpoint(target)
    assert recovered["step"] == 1
    assert checkpoint.verify_checkpoint(str(target) + ".previous")
