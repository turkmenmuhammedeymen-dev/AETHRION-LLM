"""Auditable readiness report for the complete first-party AETHRION model family.

This is a diagnostic only: it never downloads data, approves licenses, starts
training, or marks a model trained/released.
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
REQUIRED_FILES = {
    "chat": {
        "config": "configs/small.json",
        "dataset_manifest": "data/processed/corpus_manifest.json",
        "source_manifest": "data/sources.example.json",
    },
    **{
        model_id: {
            "config": f"configs/specialists/{model_id}.json",
            "dataset_manifest": f"data/specialists/manifests/{model_id}.json",
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
    candidate = Path(relative)
    if candidate.is_absolute() or "\\" in relative or "://" in relative:
        raise FamilyAuditError(f"Unsafe non-local repository path: {relative}")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise FamilyAuditError(f"Unsafe repository path: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FamilyAuditError(f"Repository path escapes root: {relative}") from exc
    return resolved


def audit_model_family(root: str | Path = ".") -> dict[str, Any]:
    """Return exact config/data/artifact readiness for all eleven model IDs."""
    repo = Path(root).resolve()
    manifest = _read_json(_local_file(repo, "models/manifest.json"), "Model manifest")
    if manifest.get("format") != "aethrion-native-model-family-v1":
        raise FamilyAuditError("Unsupported native model manifest format.")
    if manifest.get("execution_policy") != "local_only":
        raise FamilyAuditError("AETHRION model family must remain local_only.")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise FamilyAuditError("Model manifest 'models' must be an object.")

    actual = set(models)
    expected = set(EXPECTED_MODELS)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise FamilyAuditError(
            f"Model family mismatch; missing={missing}, unexpected={extra}"
        )

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
        dataset = (
            _read_json(dataset_path, f"{model_id} dataset manifest")
            if dataset_exists else {}
        )
        if model_id == "chat":
            dataset_approved = dataset.get("training_allowed") is True
            source_path = _local_file(repo, required["source_manifest"])
            source_exists = source_path.is_file()
            source_approved = False
            if source_exists:
                source = _read_json(source_path, "BASE source manifest")
                source_approved = source.get("training_allowed") is True
            data_ready = dataset_approved and source_exists and source_approved
        else:
            data_ready = (
                dataset.get("model_id") == model_id
                and dataset.get("training_allowed") is True
                and dataset.get("owner_approval") is True
                and dataset.get("privacy_reviewed") is True
                and (
                    model_id not in {"asr", "tts", "vocoder"}
                    or dataset.get("speaker_consent_reviewed") is True
                )
            )
        artifact_keys = ("checkpoint", "tokenizer", "processor", "training_report",
                         "evaluation_report", "model_card")
        artifact_presence = {}
        for key in artifact_keys:
            relative = entry.get(key)
            if not relative:
                artifact_presence[key] = key not in {"checkpoint", "training_report",
                                                     "evaluation_report", "model_card"}
                continue
            artifact_presence[key] = _local_file(repo, relative).is_file()

        rows.append({
            "model_id": model_id,
            "registry_status": entry.get("status", "missing"),
            "config_present": config_ok and bool(config),
            "dataset_manifest_present": dataset_exists,
            "training_data_approved": bool(data_ready),
            "checkpoint_present": artifact_presence["checkpoint"],
            "tokenizer_present": artifact_presence["tokenizer"],
            "processor_present": artifact_presence["processor"],
            "training_report_present": artifact_presence["training_report"],
            "evaluation_report_present": artifact_presence["evaluation_report"],
            "model_card_present": artifact_presence["model_card"],
            "training_ready": bool(config_ok and dataset_exists and data_ready),
            "release_ready": bool(
                entry.get("status") == "trained"
                and all(artifact_presence.values())
            ),
        })

    return {
        "format": "aethrion-family-readiness-audit-v1",
        "execution_policy": "local_only",
        "model_count": len(rows),
        "trained_count": sum(row["registry_status"] == "trained" for row in rows),
        "training_ready_count": sum(row["training_ready"] for row in rows),
        "release_ready_count": sum(row["release_ready"] for row in rows),
        "training_started": False,
        "automatic_approval": False,
        "models": rows,
        "notes": [
            "This audit does not download, approve, prepare, or train on any dataset.",
            "A missing specialist manifest means that model is not ready for training.",
            "A model is not released merely because its code or config exists.",
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
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
