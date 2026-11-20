"""Auditable readiness report for the complete first-party AETHRION model family.

The audit runs the same fail-closed preflight used by the training orchestrator.
It never downloads data, approves rights, starts training, or marks a model released.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_MODELS = (
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety", "vocoder",
)
SPECIALISTS = frozenset(EXPECTED_MODELS) - {"chat"}
PROCESSOR_REQUIRED = frozenset({
    "vision", "image", "asr", "tts", "3d", "research", "document", "safety"
})
REQUIRED_FILES = {
    "chat": {
        "config": "configs/small.json",
        "dataset_manifest": "data/processed/corpus_manifest.json",
        "source_manifest": "data/raw/cosmos/sources.json",
    },
    **{
        model_id: {
            "config": f"configs/specialists/{model_id}.json",
            "dataset_manifest": f"data/specialists/manifests/{model_id}.json",
            "template_manifest": f"data/specialists/manifests/templates/{model_id}.template.json",
        }
        for model_id in SPECIALISTS
    },
}


class FamilyAuditError(ValueError):
    """Invalid repository metadata prevents a trustworthy family audit."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FamilyAuditError(f"{label} is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FamilyAuditError(f"{label} must contain a JSON object: {path}")
    return value


def _local_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise FamilyAuditError("Repository artifact paths must be non-empty strings.")
    candidate = Path(relative)
    if candidate.is_absolute() or "\\\\" in relative or "://" in relative:
        raise FamilyAuditError(f"Unsafe non-local repository path: {relative}")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise FamilyAuditError(f"Unsafe repository path: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FamilyAuditError(f"Repository path escapes root: {relative}") from exc
    return resolved


def _audit_template(path: Path, model_id: str) -> tuple[str, bool]:
    """Classify a template without treating it as an approved training manifest."""
    if not path.is_file():
        return "missing", False
    try:
        payload = _read_json(path, f"{model_id} template")
        from aethrion.specialist_data import MODEL_MODALITIES
        valid = (
            payload.get("format") == "aethrion-specialist-dataset-v1"
            and payload.get("model_id") == model_id
            and payload.get("modality") == MODEL_MODALITIES[model_id]
            and payload.get("training_allowed") is False
            and payload.get("owner_approval") is False
            and payload.get("privacy_reviewed") is False
            and payload.get("sources") == []
            and isinstance(payload.get("_template_notice"), str)
            and bool(payload.get("_template_notice", "").strip())
        )
        if model_id in {"asr", "tts", "vocoder"}:
            valid = valid and payload.get("speaker_consent_reviewed") is False
    except (FamilyAuditError, KeyError, ImportError):
        valid = False
    return ("non_executable_template" if valid else "invalid_template"), valid


def audit_model_family(root: str | Path = ".") -> dict[str, Any]:
    """Audit actual train preflight, registry integrity and artifacts for all models."""
    repo = Path(root).resolve()
    manifest_path = _local_file(repo, "models/manifest.json")
    manifest = _read_json(manifest_path, "Model manifest")
    if manifest.get("format") != "aethrion-native-model-family-v1":
        raise FamilyAuditError("Unsupported native model manifest format.")
    if manifest.get("execution_policy") != "local_only":
        raise FamilyAuditError("AETHRION model family must remain local_only.")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise FamilyAuditError("Model manifest 'models' must be an object.")

    actual, expected = set(models), set(EXPECTED_MODELS)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise FamilyAuditError(f"Model family mismatch; missing={missing}, unexpected={extra}")

    # Reuse production validation instead of inferring approval from manifest booleans.
    from aethrion.train_family import preflight_model
    from aethrion.native_registry import NativeModelRegistry

    try:
        registry = NativeModelRegistry(repo)
    except Exception as exc:
        registry = None
        registry_error = f"Registry validation failed: {type(exc).__name__}: {exc}"
    else:
        registry_error = None

    rows = []
    for model_id in EXPECTED_MODELS:
        entry = models[model_id]
        if not isinstance(entry, dict):
            raise FamilyAuditError(f"Model entry must be an object: {model_id}")
        required = REQUIRED_FILES[model_id]
        config_path = _local_file(repo, required["config"])
        dataset_path = _local_file(repo, required["dataset_manifest"])
        config_ok = config_path.is_file()
        dataset_exists = dataset_path.is_file()
        config = _read_json(config_path, f"{model_id} config") if config_ok else {}

        template_status = "not_applicable"
        template_valid = False
        if model_id in SPECIALISTS:
            template_path = _local_file(repo, required["template_manifest"])
            template_status, template_valid = _audit_template(template_path, model_id)

        source_manifest_present = False
        source_count = None
        if model_id == "chat":
            source_path = _local_file(repo, required["source_manifest"])
            source_manifest_present = source_path.is_file()
            if source_manifest_present:
                source_payload = _read_json(source_path, "Chat source manifest")
                sources = source_payload.get("sources", [])
                source_count = len(sources) if isinstance(sources, list) else None
        elif dataset_exists:
            live_payload = _read_json(dataset_path, f"{model_id} dataset manifest")
            sources = live_payload.get("sources", [])
            source_count = len(sources) if isinstance(sources, list) else None
            source_manifest_present = isinstance(sources, list) and bool(sources)

        preflight_ok = False
        preflight_reason = None
        try:
            preflight_model(repo, model_id)
            preflight_ok = True
        except Exception as exc:
            preflight_reason = f"{type(exc).__name__}: {exc}"

        artifact_keys = (
            "checkpoint", "tokenizer", "processor",
            "training_report", "evaluation_report", "model_card",
        )
        required_artifacts = {
            "checkpoint", "training_report", "evaluation_report", "model_card"
        }
        if model_id != "vocoder":
            required_artifacts.add("tokenizer")
        if model_id in PROCESSOR_REQUIRED:
            required_artifacts.add("processor")

        artifact_presence: dict[str, bool] = {}
        for key in artifact_keys:
            relative = entry.get(key)
            if not relative:
                artifact_presence[key] = key not in required_artifacts
            else:
                artifact_presence[key] = _local_file(repo, relative).is_file()

        registry_status = entry.get("status", "missing")
        if registry is None:
            verified_registry_status = "registry_invalid"
        else:
            verified_registry_status = registry.status(model_id)

        rows.append({
            "model_id": model_id,
            "architecture": entry.get("architecture"),
            "registry_status": registry_status,
            "verified_registry_status": verified_registry_status,
            "config_present": config_ok and bool(config),
            "dataset_manifest_present": dataset_exists,
            "dataset_manifest_status": "present" if dataset_exists else "missing",
            "source_manifest_present": source_manifest_present,
            "source_count": source_count,
            "template_manifest_present": template_status != "missing" if model_id in SPECIALISTS else False,
            "template_manifest_valid": template_valid,
            "template_manifest_status": template_status,
            "training_data_approved": preflight_ok,
            "training_preflight_passed": preflight_ok,
            "training_ready": preflight_ok,
            "training_block_reason": preflight_reason,
            "checkpoint_present": artifact_presence["checkpoint"],
            "tokenizer_present": artifact_presence["tokenizer"],
            "processor_present": artifact_presence["processor"],
            "training_report_present": artifact_presence["training_report"],
            "evaluation_report_present": artifact_presence["evaluation_report"],
            "model_card_present": artifact_presence["model_card"],
            "required_artifacts": sorted(required_artifacts),
            "release_ready": verified_registry_status == "ready",
        })

    return {
        "format": "aethrion-family-readiness-audit-v3",
        "execution_policy": "local_only",
        "model_count": len(rows),
        "trained_count": sum(row["registry_status"] == "trained" for row in rows),
        "training_ready_count": sum(row["training_ready"] for row in rows),
        "release_ready_count": sum(row["release_ready"] for row in rows),
        "template_count": sum(row["template_manifest_valid"] for row in rows),
        "training_started": False,
        "automatic_approval": False,
        "registry_error": registry_error,
        "models": rows,
        "notes": [
            "Training readiness is based on the production train_family preflight, including local paths, source provenance, dataset hashes, tokenizer integrity and human approval gates.",
            "Specialist templates are reported separately and can never count as live datasets, approvals, or training readiness.",
            "Release readiness is based on the native registry's complete artifact, hash, benchmark, safety and human-evidence checks.",
            "The audit never downloads, approves, prepares, or trains on a dataset.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit readiness of all eleven native AETHRION models."
    )
    parser.add_argument("--root", default=".", help="AETHRION-LLM repository root.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()
    report = audit_model_family(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
