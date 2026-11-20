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
        calls += 1
        # On the second save: calls 1-2 publish the previous checkpoint and
        # its sidecar; call 3 publishes the new checkpoint; call 4 attempts
        # to publish the new sidecar. Fail precisely at that final boundary.
        if calls == 4:
            raise OSError("simulated interruption before checksum sidecar publish")
        return real_replace(source, destination)

    # Count only replacements in the second save. The first save already
    # completed before this wrapper is installed.
    monkeypatch.setattr(checkpoint.os, "replace", fail_before_checksum_publish)

    # The previous generation is copied in two replacements, then the new
    # checkpoint file is published (third), then its sidecar (fourth).
    # Verify this exact publication sequence while injecting the failure.
    with pytest.raises(OSError, match="before checksum sidecar publish"):
        checkpoint.save_checkpoint(_payload(2), target)

    monkeypatch.setattr(checkpoint.os, "replace", real_replace)

    # The new payload exists but its old sidecar no longer matches; recovery
    # must therefore come from the independently verified previous pair.
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint.verify_checkpoint(target)
    recovered = checkpoint.load_checkpoint(target)
    assert recovered["step"] == 1
    assert checkpoint.verify_checkpoint(str(target) + ".previous")
