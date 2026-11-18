"""Atomic checkpoint persistence with SHA-256 verification and recovery fallback.

Checkpoint deserialization is restricted to PyTorch's weights-only loader.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
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
    """Flush copied artifact bytes before promoting a backup generation."""
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry replacements where the platform supports it."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(path), flags)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems/platforms do not expose directory fsync.
        pass
    finally:
        os.close(descriptor)


def _write_text_fsync(path: Path, content: str, encoding: str) -> None:
    with path.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


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
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
        or not hmac.compare_digest(expected.lower(), actual)
    ):
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
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256_file(temporary)
        _write_text_fsync(sidecar_temporary, checksum + "\n", "ascii")

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
                _fsync_file(previous_tmp)
                _fsync_file(previous_sidecar_tmp)
                os.replace(previous_tmp, previous)
                os.replace(previous_sidecar_tmp, previous_sidecar)

        os.replace(temporary, target)
        os.replace(sidecar_temporary, sidecar)
        _fsync_directory(target.parent)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)


def _validate_rng_state(state: Any) -> None:
    """Reject structurally invalid RNG snapshots before choosing a generation."""
    if not isinstance(state, dict) or not {"python", "numpy", "torch"} <= set(state):
        raise ValueError("Checkpoint RNG state is incomplete.")
    py = state["python"]
    if not isinstance(py, dict) or not {"version", "state", "gauss"} <= set(py):
        raise ValueError("Checkpoint Python RNG state is invalid.")
    py_values = py["state"]
    if (
        type(py["version"]) is not int or py["version"] != 3
        or not isinstance(py_values, list) or len(py_values) != 625
        or any(type(value) is not int or value < 0 or value > 0xFFFFFFFF for value in py_values)
        or (py["gauss"] is not None and (type(py["gauss"]) not in (int, float) or not math.isfinite(float(py["gauss"]))))
    ):
        raise ValueError("Checkpoint Python RNG state is invalid.")
    ns = state["numpy"]
    if not isinstance(ns, dict) or not {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"} <= set(ns):
        raise ValueError("Checkpoint NumPy RNG state is invalid.")
    if (
        ns["algorithm"] != "MT19937" or not isinstance(ns["keys"], list) or len(ns["keys"]) != 624
        or any(type(value) is not int or value < 0 or value > 0xFFFFFFFF for value in ns["keys"])
        or type(ns["position"]) is not int or not 0 <= ns["position"] <= 624
        or type(ns["has_gauss"]) is not int or ns["has_gauss"] not in (0, 1)
        or type(ns["cached_gaussian"]) not in (int, float)
        or not math.isfinite(float(ns["cached_gaussian"]))
    ):
        raise ValueError("Checkpoint NumPy RNG state is invalid.")
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8 or torch_state.ndim != 1:
        raise ValueError("Checkpoint Torch RNG state is invalid.")
    cuda_states = state.get("cuda", [])
    if not isinstance(cuda_states, list) or any(
        not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1
        for item in cuda_states
    ):
        raise ValueError("Checkpoint CUDA RNG state is invalid.")


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
    if type(state["step"]) is not int or state["step"] < 0:
        raise ValueError("Checkpoint step is invalid.")
    for key in ("data_sha256", "tokenizer_sha256"):
        value = state[key]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError(f"Checkpoint {key} is invalid.")
    if not isinstance(state["model"], dict) or not isinstance(state["optimizer"], dict):
        raise ValueError("Checkpoint model or optimizer state is invalid.")
    if not isinstance(state["scaler"], dict) or not isinstance(state["model_config"], dict):
        raise ValueError("Checkpoint scaler or model configuration is invalid.")
    _validate_rng_state(state["rng_state"])
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
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
