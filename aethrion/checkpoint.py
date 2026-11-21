"""Atomic checkpoint persistence with SHA-256 verification and recovery fallback.

Checkpoint deserialization is restricted to PyTorch's weights-only loader.
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


def _fsync_file(path: Path) -> None:
    """Flush file contents and metadata before publishing the file."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry replacements on platforms supporting directory fsync."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    if not isinstance(payload.get("training_config"), dict) or not payload["training_config"]:
        raise ValueError("Checkpoint payload requires a non-empty training_config object.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    sidecar = _sidecar(target)
    sidecar_temporary = Path(str(sidecar) + ".tmp")
    previous = Path(str(target) + ".previous")
    previous_sidecar = _sidecar(previous)

    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        checksum = sha256_file(temporary)
        with sidecar_temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(checksum + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Preserve the last known-good generation before replacing either file.
        if target.is_file() and sidecar.is_file():
            try:
                verify_checkpoint(target)
            except ValueError:
                pass
            else:
                previous_tmp = Path(str(previous) + ".tmp")
                previous_sidecar_tmp = Path(str(previous_sidecar) + ".tmp")
                try:
                    shutil.copy2(target, previous_tmp)
                    shutil.copy2(sidecar, previous_sidecar_tmp)
                    _fsync_file(previous_tmp)
                    _fsync_file(previous_sidecar_tmp)
                    os.replace(previous_tmp, previous)
                    os.replace(previous_sidecar_tmp, previous_sidecar)
                    _fsync_directory(target.parent)
                finally:
                    previous_tmp.unlink(missing_ok=True)
                    previous_sidecar_tmp.unlink(missing_ok=True)

        os.replace(temporary, target)
        os.replace(sidecar_temporary, sidecar)
        _fsync_directory(target.parent)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)


def _load_verified_checkpoint(
    path: Path,
    map_location: str | torch.device,
) -> dict[str, Any]:
    """Verify, safely deserialize and validate one checkpoint generation."""
    verify_checkpoint(path)
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
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
    if not isinstance(state["training_config"], dict) or not state["training_config"]:
        raise ValueError("Checkpoint training_config must be a non-empty object.")
    return state


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the latest safe generation, falling back if integrity or loading fails."""
    target = Path(path)
    previous = Path(str(target) + ".previous")
    try:
        return _load_verified_checkpoint(target, map_location)
    except Exception as primary_error:
        try:
            recovered = _load_verified_checkpoint(previous, map_location)
        except Exception:
            raise primary_error
        warnings.warn(
            f"Latest checkpoint could not be verified and safely loaded; recovering from {previous}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return recovered



def reconcile_metrics_jsonl(path: str | Path, checkpoint_step: int) -> int:
    """Discard malformed, duplicate, out-of-order, or uncommitted metrics after recovery.

    A metrics row is durable before its corresponding checkpoint is published.
    If a process dies in that window, the row may describe a step that is not in
    the verified checkpoint. Resume must remove that row before appending again.
    Returns the number of discarded rows.
    """
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise ValueError("checkpoint_step must be a non-negative integer.")
    target = Path(path)
    if not target.is_file():
        return 0

    raw_metrics = target.read_text(encoding="utf-8")
    retained: list[str] = []
    last_step = -1
    discarded = 0
    for line in raw_metrics.splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            discarded += 1
            continue
        if not isinstance(record, dict):
            discarded += 1
            continue
        step = record.get("step")
        if type(step) is not int or step < 0 or step > checkpoint_step or step <= last_step:
            discarded += 1
            continue
        retained.append(json.dumps(record, ensure_ascii=False) + "\n")
        last_step = step

    needs_rewrite = discarded > 0 or bool(raw_metrics and not raw_metrics.endswith("\n"))
    if needs_rewrite:
        temporary = Path(str(target) + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.writelines(retained)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return discarded


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
