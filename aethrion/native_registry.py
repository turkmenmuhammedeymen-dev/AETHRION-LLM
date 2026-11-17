"""Strict local-only registry for independently trained AETHRION model artifacts."""
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
    "chat", "code", "vision", "image", "asr", "tts", "vocoder", "3d",
    "research", "document", "safety",
})
PROCESSOR_REQUIRED = frozenset({"vision", "image", "asr", "tts", "3d", "document"})
_ALLOWED_ENTRY_KEYS = frozenset({
    "architecture", "model_config",
    "checkpoint", "tokenizer", "processor",
    "checkpoint_sha256", "tokenizer_sha256", "processor_sha256",
    "training_report", "training_report_sha256",
    "evaluation_report", "evaluation_report_sha256",
    "dataset_sha256", "release_approval", "quality_gate_passed", "safety_gate_passed",
    "model_card", "model_card_sha256", "status",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativeModelError(RuntimeError):
    """Base error for native model registry failures."""


class NativeModelUnavailable(NativeModelError):
    """Raised when a first-party model is not trained, installed or released."""


class NativeModelIntegrityError(NativeModelError):
    """Raised when a model artifact or release record fails verification."""


@dataclass(frozen=True)
class NativeModelArtifacts:
    capability: str
    architecture: str
    checkpoint: Path
    model_config: dict[str, Any]
    tokenizer: Path | None = None
    processor: Path | None = None
    training_report: Path | None = None
    evaluation_report: Path | None = None
    model_card: Path | None = None


class NativeModelRegistry:
    """Resolve only local, hash-verified, independently evaluated model releases."""

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
    def _validate_local_relative_path(capability: str, key: str, value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            raise NativeModelError(f"Invalid {key} path for {capability}.")
        if (
            value.startswith("/")
            or chr(92) in value
            or "://" in value
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value)
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise NativeModelError(f"{key} must be a safe relative local path for {capability}.")

    @staticmethod
    def _validate_digest(capability: str, key: str, value: Any) -> None:
        if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
            raise NativeModelError(f"Invalid SHA-256 value for {capability}.{key}.")

    @classmethod
    def _validate_entry(cls, capability: str, entry: Any) -> None:
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
        for key in ("checkpoint", "tokenizer", "processor", "training_report",
                    "evaluation_report", "model_card"):
            if key in entry:
                cls._validate_local_relative_path(capability, key, entry[key])
        for key in ("checkpoint_sha256", "tokenizer_sha256", "processor_sha256",
                    "training_report_sha256", "evaluation_report_sha256",
                    "model_card_sha256", "dataset_sha256"):
            if key in entry:
                cls._validate_digest(capability, key, entry[key])
        config = entry.get("model_config")
        if config is not None and (not isinstance(config, dict) or not config):
            raise NativeModelError(f"{capability}.model_config must be a non-empty object.")
        if entry["status"] == "trained":
            required_paths = ("checkpoint", "tokenizer", "training_report", "evaluation_report", "model_card")
            required_hashes = ("checkpoint_sha256", "tokenizer_sha256", "training_report_sha256",
                               "evaluation_report_sha256", "model_card_sha256", "dataset_sha256")
            for key in required_paths:
                if not entry.get(key):
                    raise NativeModelError(f"Trained {capability} model requires {key}.")
            for key in required_hashes:
                if not entry.get(key):
                    raise NativeModelError(f"Trained {capability} model requires {key}.")
            if not isinstance(config, dict) or not config:
                raise NativeModelError(f"Trained {capability} model requires model_config.")
            if capability in PROCESSOR_REQUIRED and not entry.get("processor"):
                raise NativeModelError(f"Trained {capability} model requires its processor.")
            for flag in ("release_approval", "quality_gate_passed", "safety_gate_passed"):
                if type(entry.get(flag)) is not bool or entry[flag] is not True:
                    raise NativeModelError(f"Trained {capability} model requires {flag}=true.")

    def _safe_path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or chr(92) in str(relative):
            raise NativeModelError("Absolute paths and Windows path separators are forbidden.")
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
        except NativeModelError:
            return "release_gate_failed"
        return "ready"

    def _verified_file(self, capability: str, entry: dict[str, Any],
                       path_key: str, hash_key: str) -> Path:
        relative = entry.get(path_key)
        if not relative:
            raise NativeModelUnavailable(f"Local {capability} {path_key} is not registered.")
        path = self._safe_path(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise NativeModelUnavailable(f"Local {capability} {path_key} artifact is missing or empty.")
        expected = entry.get(hash_key)
        if not expected or self._digest(path) != expected.lower():
            raise NativeModelIntegrityError(f"SHA-256 verification failed for {capability} {path_key}.")
        return path

    def resolve(self, capability: str) -> NativeModelArtifacts:
        entry = self._models.get(capability)
        if entry is None:
            raise NativeModelUnavailable(f"No native AETHRION model is registered for {capability}.")
        if entry["status"] != "trained":
            raise NativeModelUnavailable(f"AETHRION {capability} model is not trained and released.")
        checkpoint = self._verified_file(capability, entry, "checkpoint", "checkpoint_sha256")
        tokenizer = self._verified_file(capability, entry, "tokenizer", "tokenizer_sha256")
        processor = None
        if entry.get("processor"):
            processor = self._verified_file(capability, entry, "processor", "processor_sha256")
        training_report = self._verified_file(
            capability, entry, "training_report", "training_report_sha256"
        )
        evaluation_report = self._verified_file(
            capability, entry, "evaluation_report", "evaluation_report_sha256"
        )
        model_card = self._verified_file(capability, entry, "model_card", "model_card_sha256")
        try:
            training = json.loads(training_report.read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NativeModelIntegrityError(f"{capability} release reports are invalid JSON.") from exc
        if training.get("model_id") != capability or training.get("dataset_sha256") != entry["dataset_sha256"]:
            raise NativeModelIntegrityError(f"{capability} training report does not match its dataset.")
        if evaluation.get("model_id") != capability or evaluation.get("dataset_sha256") != entry["dataset_sha256"]:
            raise NativeModelIntegrityError(f"{capability} evaluation report does not match its dataset.")
        if evaluation.get("status") != "approved_for_release":
            raise NativeModelUnavailable(f"{capability} evaluation report is not approved for release.")
        return NativeModelArtifacts(
            capability=capability,
            architecture=entry["architecture"],
            checkpoint=checkpoint,
            model_config=entry["model_config"],
            tokenizer=tokenizer,
            processor=processor,
            training_report=training_report,
            evaluation_report=evaluation_report,
            model_card=model_card,
        )
