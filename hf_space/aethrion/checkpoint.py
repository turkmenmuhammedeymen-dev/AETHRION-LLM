"""Atomic checkpoint persistence with SHA-256 verification and recovery fallback.

Only load checkpoints produced by this project or another trusted source:
PyTorch optimizer/RNG checkpoints use pickle-backed serialization.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def verify_checkpoint(path: str | Path) -> str:
    """Verify a checkpoint against its adjacent checksum file."""
    target = Path(path)
    sidecar = _sidecar(target)
    if not target.is_file():
        raise ValueError(f"Checkpoint does not exist: {target}")
    if not sidecar.is_file():
        raise ValueError(f"Checkpoint integrity file is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if not fields:
        raise ValueError(f"Checkpoint integrity file is empty: {sidecar}")
    expected = fields[0]
    actual = sha256_file(target)
    if len(expected) != 64 or not hmac.compare_digest(expected.lower(), actual):
        raise ValueError(f"Checkpoint SHA-256 verification failed: {target}")
    return actual


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> str:
    """Write a checkpoint atomically; retain one verified previous generation."""
    if not isinstance(payload, dict) or payload.get("format_version") != 4:
        raise ValueError("Checkpoint payload must use format_version=4.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    sidecar = _sidecar(target)
    sidecar_temporary = Path(str(sidecar) + ".tmp")
    previous = Path(str(target) + ".previous")
    previous_sidecar = _sidecar(previous)

    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256_file(temporary)
        sidecar_temporary.write_text(checksum + "\n", encoding="ascii")

        # Preserve the last known-good generation before replacing either file.
        if target.is_file() and sidecar.is_file():
            try:
                verify_checkpoint(target)
            except ValueError:
                pass
            else:
                previous_tmp = Path(str(previous) + ".tmp")
                previous_sidecar_tmp = Path(str(previous_sidecar) + ".tmp")
                shutil.copy2(target, previous_tmp)
                shutil.copy2(sidecar, previous_sidecar_tmp)
                os.replace(previous_tmp, previous)
                os.replace(previous_sidecar_tmp, previous_sidecar)

        os.replace(temporary, target)
        os.replace(sidecar_temporary, sidecar)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only after checksum validation; recover from one corrupt latest save."""
    target = Path(path)
    load_path = target
    try:
        verify_checkpoint(target)
    except ValueError as primary_error:
        previous = Path(str(target) + ".previous")
        try:
            verify_checkpoint(previous)
        except ValueError:
            raise primary_error
        warnings.warn(
            f"Latest checkpoint failed integrity verification; recovering from {previous}.",
            RuntimeWarning,
            stacklevel=2,
        )
        load_path = previous

    try:
        state = torch.load(load_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Checkpoint could not be safely deserialized; unsafe pickle objects are not accepted."
        ) from exc
    if not isinstance(state, dict) or state.get("format_version") != 4:
        raise ValueError("Unsupported or invalid AETHRION checkpoint format.")
    required = {
        "step", "model", "optimizer", "scaler", "model_config",
        "data_sha256", "tokenizer_sha256", "rng_state", "training_config",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError("Checkpoint is missing required fields: " + ", ".join(missing))
    return state


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
