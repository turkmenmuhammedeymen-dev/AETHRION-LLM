"""Strict local-only registry for first-party AETHRION model artifacts.

This module performs no network access and never downloads or substitutes models.
A model is resolvable only when its status is trained and all declared local
artifacts pass path and SHA-256 verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


FORMAT = "aethrion-native-model-family-v1"
POLICY = "local_only"
CAPABILITIES = frozenset({
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety",
})
_ALLOWED_ENTRY_KEYS = frozenset({
    "architecture", "checkpoint", "tokenizer", "processor",
    "checkpoint_sha256", "tokenizer_sha256", "processor_sha256", "status",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativeModelError(RuntimeError):
    """Base error for local native-model registry failures."""


class NativeModelUnavailable(NativeModelError):
    """Raised when a first-party model is not trained or not installed."""


class NativeModelIntegrityError(NativeModelError):
    """Raised when a local model artifact fails integrity verification."""


@dataclass(frozen=True)
class NativeModelArtifacts:
    capability: str
    architecture: str
    checkpoint: Path
    tokenizer: Path | None = None
    processor: Path | None = None


class NativeModelRegistry:
    """Resolve only explicitly trained, hash-verified local model artifacts."""

    def __init__(self, root: str | Path, manifest: str | Path = "models/manifest.json"):
        self.root = Path(root).resolve()
        manifest_path = self._safe_path(manifest)
        if not manifest_path.is_file():
            raise NativeModelUnavailable(f"Native model manifest not found: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NativeModelError("Native model manifest is unreadable or invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise NativeModelError("Native model manifest must be a JSON object.")
        if payload.get("format") != FORMAT or payload.get("execution_policy") != POLICY:
            raise NativeModelError("Unsupported manifest format or non-local execution policy.")
        models = payload.get("models")
        if not isinstance(models, dict):
            raise NativeModelError("Manifest models must be an object.")
        unknown = set(models) - CAPABILITIES
        if unknown:
            raise NativeModelError("Unknown model capabilities: " + ", ".join(sorted(unknown)))
        self._models: dict[str, dict[str, Any]] = {}
        for capability, entry in models.items():
            self._validate_entry(capability, entry)
            self._models[capability] = entry

    @staticmethod
    def _validate_entry(capability: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            raise NativeModelError(f"Model entry for {capability} must be an object.")
        extra = set(entry) - _ALLOWED_ENTRY_KEYS
        if extra:
            raise NativeModelError(
                f"Unsupported fields in {capability} model entry: " + ", ".join(sorted(extra))
            )
        if entry.get("status") not in {"not_trained", "trained", "blocked"}:
            raise NativeModelError(f"Invalid training status for {capability}.")
        architecture = entry.get("architecture")
        if not isinstance(architecture, str) or not architecture.strip() or ":" not in architecture:
            raise NativeModelError(f"Missing local architecture identifier for {capability}.")
        for key in ("checkpoint", "tokenizer", "processor"):
            value = entry.get(key)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise NativeModelError(f"Invalid {key} path for {capability}.")
                normalized = value
                if (
                    normalized.startswith("/")
                    or chr(92) in normalized
                    or "://" in normalized
                    or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
                    or any(part == ".." for part in normalized.split("/"))
                ):
                    raise NativeModelError(f"{key} must be a safe relative local path for {capability}.")
        for key in ("checkpoint_sha256", "tokenizer_sha256", "processor_sha256"):
            value = entry.get(key)
            if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value.lower())):
                raise NativeModelError(f"Invalid SHA-256 value for {capability}.{key}.")
        if entry["status"] == "trained":
            if not entry.get("checkpoint") or not entry.get("checkpoint_sha256"):
                raise NativeModelError(f"Trained {capability} model needs a checkpoint and SHA-256.")
            if entry.get("tokenizer") and not entry.get("tokenizer_sha256"):
                raise NativeModelError(f"{capability} tokenizer needs a SHA-256.")
            if entry.get("processor") and not entry.get("processor_sha256"):
                raise NativeModelError(f"{capability} processor needs a SHA-256.")

    def _safe_path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise NativeModelError("Absolute artifact paths are forbidden.")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise NativeModelError("Artifact path escapes the model root.") from exc
        return resolved

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def status(self, capability: str) -> str:
        entry = self._models.get(capability)
        if entry is None:
            return "not_configured"
        if entry["status"] != "trained":
            return entry["status"]
        try:
            self.resolve(capability)
        except NativeModelUnavailable:
            return "missing_artifact"
        except NativeModelIntegrityError:
            return "integrity_failed"
        return "ready"

    def resolve(self, capability: str) -> NativeModelArtifacts:
        entry = self._models.get(capability)
        if entry is None:
            raise NativeModelUnavailable(f"No native AETHRION model is registered for {capability}.")
        if entry["status"] != "trained":
            raise NativeModelUnavailable(f"AETHRION {capability} model is not trained and released.")
        resolved: dict[str, Path | None] = {}
        for kind in ("checkpoint", "tokenizer", "processor"):
            relative = entry.get(kind)
            if relative is None:
                resolved[kind] = None
                continue
            path = self._safe_path(relative)
            if not path.is_file() or path.stat().st_size == 0:
                raise NativeModelUnavailable(f"Local {capability} {kind} artifact is missing or empty.")
            expected = entry.get(kind + "_sha256")
            actual = self._digest(path)
            if not expected or actual != expected.lower():
                raise NativeModelIntegrityError(f"SHA-256 verification failed for {capability} {kind}.")
            resolved[kind] = path
        return NativeModelArtifacts(
            capability=capability,
            architecture=entry["architecture"],
            checkpoint=resolved["checkpoint"],
            tokenizer=resolved["tokenizer"],
            processor=resolved["processor"],
        )
