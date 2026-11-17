"""Fail-closed sequential training orchestrator for all eleven native models.

The default mode is preflight-only. Training starts only with --execute and
only after the model-specific data validator accepts the exact local dataset.
No third-party model weights or inference APIs are used.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_IDS = (
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety", "vocoder",
)
SPECIALISTS = frozenset(MODEL_IDS) - {"chat"}


class FamilyTrainingError(ValueError):
    """A model cannot enter the training queue."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FamilyTrainingError(f"Training config is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise FamilyTrainingError(f"Training config must be a JSON object: {path}")
    return value


def preflight_model(root: str | Path, model_id: str) -> dict[str, Any]:
    """Validate the real data gate without initializing a model or optimizer."""
    repo = Path(root).resolve()
    if model_id not in MODEL_IDS:
        raise FamilyTrainingError(f"Unknown AETHRION model ID: {model_id}")

    if model_id == "chat":
        config_path = repo / "configs" / "small.json"
        config = _read_config(config_path)
        dataset_manifest = repo / "data" / "processed" / "corpus_manifest.json"
        source_manifest = repo / "data" / "raw" / "cosmos" / "sources.json"
        source_root = repo / "data" / "raw" / "cosmos"
        corpus = (repo / str(config.get("data_path", ""))).resolve()
        tokenizer = (repo / str(config.get("tokenizer_path", ""))).resolve()
        report = repo / "data" / "processed" / "corpus-report.json"
        source_review = repo / "data" / "processed" / "privacy_review" / "source_review.csv"
        from aethrion.prepare_data import validate_training_approval
        validate_training_approval(
            dataset_manifest, source_manifest, source_root, corpus, report,
            source_review if source_review.is_file() else None,
        )
        if not tokenizer.is_file():
            raise FamilyTrainingError(f"BASE tokenizer is missing: {tokenizer}")
        output_dir = (repo / str(config.get("output_dir", "artifacts"))).resolve()
        command = [
            sys.executable, "-m", "aethrion.train",
            "--config", str(config_path.relative_to(repo)),
            "--dataset-manifest", str(dataset_manifest.relative_to(repo)),
            "--source-manifest", str(source_manifest.relative_to(repo)),
            "--source-root", str(source_root.relative_to(repo)),
            "--corpus-report", str(report.relative_to(repo)),
        ]
        if source_review.is_file():
            command.extend(["--source-review", str(source_review.relative_to(repo))])
        checkpoint = output_dir / "checkpoint_last.pt"
    else:
        config_path = repo / "configs" / "specialists" / f"{model_id}.json"
        config = _read_config(config_path)
        manifest_path = repo / "data" / "specialists" / "manifests" / f"{model_id}.json"
        from aethrion.specialist_data import validate_specialist_dataset
        approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
        if approval.get("model_id") != model_id:
            raise FamilyTrainingError("Approved dataset manifest targets a different model.")
        output_dir = (repo / str(config.get("output_dir", ""))).resolve()
        command = [
            sys.executable, "-m", "aethrion.specialist_train",
            "--config", str(config_path.relative_to(repo)),
            "--manifest", str(manifest_path.relative_to(repo)),
        ]
        checkpoint = output_dir / "checkpoint_last.pt"

    try:
        output_dir.relative_to(repo)
    except ValueError as exc:
        raise FamilyTrainingError("Configured output directory escapes the repository root.") from exc
    if checkpoint.is_file():
        command.extend(["--resume", str(checkpoint.relative_to(repo))])

    return {
        "model_id": model_id,
        "ready": True,
        "command": command,
        "checkpoint": str(checkpoint.relative_to(repo)),
        "output_dir": str(output_dir.relative_to(repo)),
    }



def _training_outcome(output_dir: Path, model_id: str, returncode: int) -> str:
    """Derive status from the trainer's durable report, never from exit code alone."""
    if returncode != 0:
        return "training_failed_not_released"
    report_path = output_dir / "training_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "training_report_invalid_not_released"
        status = report.get("status")
        if status == "training_complete_not_released":
            return "training_completed_not_released"
        if status == "training_interrupted_not_released":
            return "training_interrupted_not_released"
        if status == "instruction_tuning_complete_not_released":
            return "instruction_tuning_completed_not_released"
        return "trainer_report_unrecognized_not_released"
    if model_id == "chat":
        progress_path = output_dir / "progress.json"
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return "training_progress_invalid_not_released"
            if progress.get("status") == "interrupted":
                return "training_interrupted_not_released"
    return "trainer_exited_without_verified_report"


def run_family(
    root: str | Path = ".",
    *,
    selected: list[str] | None = None,
    execute: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Preflight or sequentially execute every selected, fully approved model."""
    repo = Path(root).resolve()
    model_ids = selected or list(MODEL_IDS)
    if len(set(model_ids)) != len(model_ids):
        raise FamilyTrainingError("Duplicate model IDs are not allowed.")
    unknown = sorted(set(model_ids) - set(MODEL_IDS))
    if unknown:
        raise FamilyTrainingError("Unknown model IDs: " + ", ".join(unknown))

    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        try:
            plan = preflight_model(repo, model_id)
            command = list(plan["command"])
            if python_executable:
                command[0] = python_executable
            if not execute:
                rows.append({**plan, "status": "ready_not_started"})
                continue
            completed = subprocess.run(command, cwd=repo, check=False)
            rows.append({
                **plan,
                "command": command,
                "status": _training_outcome(
                    repo / plan["output_dir"], model_id, int(completed.returncode)
                ),
                "returncode": int(completed.returncode),
            })
        except Exception as exc:
            rows.append({
                "model_id": model_id,
                "ready": False,
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            })

    started = execute and any(
        row.get("status") in {
            "training_completed_not_released", "training_failed_not_released",
            "training_interrupted_not_released", "trainer_exited_without_verified_report",
            "training_report_invalid_not_released", "training_progress_invalid_not_released",
            "trainer_report_unrecognized_not_released",
            "instruction_tuning_completed_not_released",
        }
        for row in rows
    )
    return {
        "format": "aethrion-family-training-run-v1",
        "execution_policy": "local_only",
        "model_count": len(rows),
        "ready_count": sum(row.get("ready") is True for row in rows),
        "blocked_count": sum(row.get("status") == "blocked" for row in rows),
        "training_started": bool(started),
        "released_count": 0,
        "execute_requested": bool(execute),
        "models": rows,
        "note": (
            "A successful training process is not a release. Independent held-out "
            "evaluation, safety review, artifact integrity and human release approval remain mandatory."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or sequentially train approved first-party AETHRION models."
    )
    parser.add_argument("--root", default=".", help="AETHRION-LLM repository root.")
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=None)
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually run each model trainer after its complete data gate passes.",
    )
    parser.add_argument("--python", default=None, help="Python executable for child trainers.")
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()
    report = run_family(
        args.root, selected=args.models, execute=args.execute,
        python_executable=args.python,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(target) + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(target)
    return 0 if report["blocked_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
