"""Strict local-only registry for independently trained AETHRION model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


FORMAT = "aethrion-native-model-family-v1"
POLICY = "local_only"
CAPABILITIES = frozenset({
    "chat", "code", "vision", "image", "asr", "tts", "vocoder", "3d",
    "research", "document", "safety",
})
PROCESSOR_REQUIRED = frozenset({"vision", "image", "asr", "tts", "3d", "research", "document", "safety"})
REQUIRED_BENCHMARKS = {
    "chat": {"held_out_lm_loss", "instruction_following"},
    "code": {"unit_test_pass_rate", "security_regression"},
    "vision": {"recall_at_1", "image_text_alignment"},
    "image": {"fid_or_equivalent", "prompt_alignment", "image_safety"},
    "asr": {"wer", "cer"},
    "tts": {"mos", "intelligibility"},
    "vocoder": {"mos", "spectral_distortion"},
    "3d": {"chamfer_distance", "point_cloud_validity"},
    "research": {"recall_at_5", "citation_precision"},
    "document": {"field_f1", "extractive_qa_em"},
    "safety": {"false_negative_rate", "calibration_ece"},
}
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
            required_paths = ("checkpoint", "training_report", "evaluation_report", "model_card")
            required_hashes = ("checkpoint_sha256", "training_report_sha256",
                               "evaluation_report_sha256", "model_card_sha256", "dataset_sha256")
            if capability != "vocoder":
                required_paths += ("tokenizer",)
                required_hashes += ("tokenizer_sha256",)
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
        tokenizer = None
        if entry.get("tokenizer"):
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
        if training.get("status") not in {
            "training_complete_not_released", "instruction_tuning_complete_not_released"
        }:
            raise NativeModelUnavailable(f"{capability} training report is not a completed native training run.")
        if training.get("checkpoint_sha256") != self._digest(checkpoint):
            raise NativeModelIntegrityError(f"{capability} checkpoint hash does not match the training report.")
        if evaluation.get("model_id") != capability or evaluation.get("dataset_sha256") != entry["dataset_sha256"]:
            raise NativeModelIntegrityError(f"{capability} evaluation report does not match its dataset.")
        if evaluation.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
            raise NativeModelIntegrityError(f"{capability} evaluation report does not match its checkpoint.")
        if evaluation.get("status") != "approved_for_release":
            raise NativeModelUnavailable(f"{capability} evaluation report is not approved for release.")
        for flag in ("quality_gate_passed", "safety_gate_passed", "human_reviewed"):
            if type(evaluation.get(flag)) is not bool or evaluation[flag] is not True:
                raise NativeModelUnavailable(
                    f"{capability} evaluation report requires {flag}=true."
                )
        for key in ("reviewer", "reviewed_at", "approval_evidence_path",
                    "approval_evidence_sha256"):
            if not isinstance(evaluation.get(key), str) or not evaluation[key].strip():
                raise NativeModelUnavailable(
                    f"{capability} evaluation report is missing human review evidence: {key}."
                )
        benchmark_results = evaluation.get("benchmarks")
        required_benchmarks = REQUIRED_BENCHMARKS.get(capability, set())
        if not isinstance(benchmark_results, dict) or not required_benchmarks.issubset(benchmark_results):
            missing = sorted(required_benchmarks - set(benchmark_results or {}))
            raise NativeModelUnavailable(
                f"{capability} evaluation report is missing required benchmarks: {', '.join(missing)}."
            )
        for benchmark_name, result in benchmark_results.items():
            if not isinstance(benchmark_name, str) or not benchmark_name.strip():
                raise NativeModelIntegrityError("Benchmark names must be non-empty strings.")
            if not isinstance(result, dict) or not result:
                raise NativeModelIntegrityError(f"Invalid benchmark record: {benchmark_name}.")
            value = result.get("value")
            if (
                result.get("passed") is not True
                or not isinstance(result.get("metric"), str)
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not isinstance(result.get("threshold"), (int, float))
                or isinstance(result.get("threshold"), bool)
                or not math.isfinite(float(result["threshold"]))
                or not isinstance(result.get("evidence"), str)
                or not result["evidence"].strip()
            ):
                raise NativeModelUnavailable(
                    f"Benchmark {benchmark_name} has not passed with a finite numeric measurement."
                )
        evidence_path = self._safe_path(evaluation["approval_evidence_path"])
        if not evidence_path.is_file() or self._digest(evidence_path) != evaluation["approval_evidence_sha256"].lower():
            raise NativeModelIntegrityError(f"{capability} human release evidence is missing or hash-mismatched.")
        card_text = model_card.read_text(encoding="utf-8")
        for heading in ("## Intended use", "## Training data", "## Evaluation", "## Limitations"):
            if heading not in card_text:
                raise NativeModelUnavailable(f"{capability} model card must include '{heading}'.")
        if capability == "tts" and self.status("vocoder") != "ready":
            raise NativeModelUnavailable("AETHRION-TTS requires a separately trained and released AETHRION-VOCODER.")
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
