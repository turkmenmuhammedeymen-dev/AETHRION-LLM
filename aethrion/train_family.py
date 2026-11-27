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


def _resolve_repo_input(repo: Path, raw_value: Any, field_name: str) -> Path:
    """Resolve an existing regular input file confined to the repository."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise FamilyTrainingError(f"A non-empty relative {field_name} is required.")
    configured = Path(raw_value)
    if (
        configured.is_absolute()
        or "\\" in raw_value
        or (len(raw_value) >= 2 and raw_value[1] == ":")
        or any(part in {"", ".", ".."} for part in raw_value.split("/"))
    ):
        raise FamilyTrainingError(f"{field_name} must be a safe relative path.")
    resolved = (repo / configured).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FamilyTrainingError(f"Configured {field_name} escapes the repository root.") from exc
    if not resolved.is_file():
        raise FamilyTrainingError(f"Configured {field_name} does not exist as a regular file: {raw_value}")
    return resolved

def _resolve_output_dir(repo: Path, raw_value: Any) -> Path:
    """Require an explicit, non-root output folder contained inside the repository."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise FamilyTrainingError("A non-empty relative output_dir is required.")
    configured = Path(raw_value)
    if configured.is_absolute() or any(part == ".." for part in configured.parts):
        raise FamilyTrainingError("output_dir must be a safe relative path.")
    resolved = (repo / configured).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FamilyTrainingError("Configured output directory escapes the repository root.") from exc
    if resolved == repo:
        raise FamilyTrainingError("output_dir cannot be the repository root.")
    return resolved


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
        corpus = _resolve_repo_input(repo, config.get("data_path"), "data_path")
        tokenizer = _resolve_repo_input(repo, config.get("tokenizer_path"), "tokenizer_path")
        report = repo / "data" / "processed" / "corpus-report.json"
        source_review = repo / "data" / "processed" / "privacy_review" / "source_review.csv"
        from aethrion.tokenizer import validate_tokenizer_provenance
        validate_tokenizer_provenance(
            tokenizer, corpus, dataset_manifest, source_manifest, source_root, report,
            source_review if source_review.is_file() else None,
        )
        output_dir = _resolve_output_dir(repo, config.get("output_dir", "artifacts"))
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
        if config.get("model_id") != model_id:
            raise FamilyTrainingError("Training config model_id does not match the selected specialist.")
        if not isinstance(config.get("model_config"), dict) or not config["model_config"]:
            raise FamilyTrainingError("A non-empty model_config object is required.")
        output_dir = _resolve_output_dir(repo, config.get("output_dir"))
        manifest_path = repo / "data" / "specialists" / "manifests" / f"{model_id}.json"
        from aethrion.specialist_data import validate_specialist_dataset
        approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
        if approval.get("model_id") != model_id:
            raise FamilyTrainingError("Approved dataset manifest targets a different model.")
        command = [
            sys.executable, "-m", "aethrion.specialist_train",
            "--config", str(config_path.relative_to(repo)),
            "--manifest", str(manifest_path.relative_to(repo)),
        ]
        checkpoint = output_dir / "checkpoint_last.pt"

    if checkpoint.is_file():
        command.extend(["--resume", str(checkpoint.relative_to(repo))])

    return {
        "model_id": model_id,
        "ready": True,
        "command": command,
        "checkpoint": str(checkpoint.relative_to(repo)),
        "output_dir": str(output_dir.relative_to(repo)),
    }



def _run_forwarding_stop_signals(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a trainer and forward SIGTERM/SIGINT so it can checkpoint before pausing."""
    import signal
    child = subprocess.Popen(command, cwd=cwd)
    previous = {}

    def _forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[stop_signal] = signal.signal(stop_signal, _forward)
        except ValueError:  # not in the main thread
            pass
    try:
        returncode = child.wait()
    finally:
        for stop_signal, handler in previous.items():
            signal.signal(stop_signal, handler)
    return subprocess.CompletedProcess(command, returncode)


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
    session_steps: int | None = None,
) -> dict[str, Any]:
    """Preflight or sequentially execute every selected, fully approved model."""
    repo = Path(root).resolve()
    if session_steps is not None and (type(session_steps) is not int or session_steps < 1):
        raise FamilyTrainingError("session_steps must be a positive integer.")
    if type(execute) is not bool:
        raise FamilyTrainingError("execute must be an explicit boolean.")
    if selected is None:
        model_ids = list(MODEL_IDS)
    else:
        if not isinstance(selected, list):
            raise FamilyTrainingError("selected must be a list of model IDs or None.")
        if not selected:
            raise FamilyTrainingError(
                "An explicit empty model selection is invalid; use selected=None to select all eleven models."
            )
        if any(not isinstance(model_id, str) or not model_id for model_id in selected):
            raise FamilyTrainingError("Every selected model ID must be a non-empty string.")
        model_ids = list(selected)
    if python_executable is not None and (
        not isinstance(python_executable, str) or not python_executable.strip()
    ):
        raise FamilyTrainingError("python_executable must be a non-empty string when provided.")
    if len(set(model_ids)) != len(model_ids):
        raise FamilyTrainingError("Duplicate model IDs are not allowed.")
    unknown = sorted(set(model_ids) - set(MODEL_IDS))
    if unknown:
        raise FamilyTrainingError("Unknown model IDs: " + ", ".join(unknown))

    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any] | None] = []
    # Phase 1: validate every selected model before starting any trainer.
    # This prevents a partially approved family request from training a subset
    # merely because its blocked member appears later in the list.
    for model_id in model_ids:
        try:
            plan = preflight_model(repo, model_id)
            command = list(plan["command"])
            if python_executable:
                command[0] = python_executable
            if session_steps is not None:
                command.extend(["--session-steps", str(session_steps)])
            plans.append({**plan, "command": command})
            rows.append({**plan, "command": command, "status": "ready_not_started"})
        except Exception as exc:
            plans.append(None)
            rows.append({
                "model_id": model_id,
                "ready": False,
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            })

    family_preflight_blocked = any(plan is None for plan in plans)
    if execute and family_preflight_blocked:
        # No subprocess is allowed to start until the entire requested family
        # has passed preflight. Keep the individual ready/blocked diagnostics.
        for row in rows:
            if row.get("ready") is True:
                row["status"] = "not_started_family_preflight_blocked"
                row["reason"] = "At least one selected model failed preflight; no model was started."
    elif execute:
        # Phase 2: all selected models are ready; now execute in the requested order.
        for row, plan in zip(rows, plans):
            assert plan is not None
            try:
                completed = _run_forwarding_stop_signals(plan["command"], repo)
                row["status"] = _training_outcome(
                    repo / plan["output_dir"], row["model_id"], int(completed.returncode)
                )
                row["returncode"] = int(completed.returncode)
            except Exception as exc:
                row["status"] = "training_launch_failed_not_released"
                row["reason"] = f"{type(exc).__name__}: {exc}"

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
        "execution_blocked_by_family_preflight": bool(execute and family_preflight_blocked),
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
    parser.add_argument("--session-steps", type=int, default=None,
                        help="Maximum optimizer steps per model in this invocation; saves and pauses for resume.")
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()
    report = run_family(
        args.root, selected=args.models, execute=args.execute,
        python_executable=args.python, session_steps=args.session_steps,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(target) + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(target)
    if report["blocked_count"]:
        return 2
    failed_statuses = {
        "training_failed_not_released",
        "training_launch_failed_not_released",
        "training_report_invalid_not_released",
        "training_progress_invalid_not_released",
        "trainer_exited_without_verified_report",
        "trainer_report_unrecognized_not_released",
    }
    execution_failed_count = sum(row.get("status") in failed_statuses for row in report["models"])
    report["execution_failed_count"] = execution_failed_count
    if args.execute and execution_failed_count:
        print("TRAINING EXECUTION FAILED: one or more trainers did not produce a verified report.", file=sys.stderr)
        # Rewrite the report with the failure count so automation can audit the result.
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered, file=sys.stderr)
        if args.report:
            target = Path(args.report)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(target) + ".tmp")
            temporary.write_text(rendered + "\\n", encoding="utf-8")
            temporary.replace(target)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
