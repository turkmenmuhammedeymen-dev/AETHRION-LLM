"""Fail-closed sequential training orchestrator for all eleven native models.

The default mode is preflight-only. Training starts only with --execute and
only after the model-specific data validator accepts the exact local dataset.
No third-party model weights or inference APIs are used.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aethrion.checkpoint import write_json_atomic
from aethrion.training_config import require_finite_number, require_int, validate_numeric_tree

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
    if (
        configured.is_absolute()
        or "\\" in raw_value
        or ":" in raw_value
        or any(part in {"", ".", ".."} for part in raw_value.split("/"))
    ):
        raise FamilyTrainingError("output_dir must be a safe relative path.")
    resolved = (repo / configured).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FamilyTrainingError("Configured output directory escapes the repository root.") from exc
    if resolved == repo:
        raise FamilyTrainingError("output_dir cannot be the repository root.")
    return resolved


SPECIALIST_MODEL_CONFIG_KEYS = {
    "code": ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"),
    "vision": ("image_size", "patch_size", "vocab_size", "max_text_length", "width", "layers", "heads", "projection_dim"),
    "image": ("vocab_size", "image_size", "text_length", "width", "time_dim", "diffusion_steps"),
    "asr": ("vocab_size", "n_mels", "width", "layers", "heads", "blank_id"),
    "tts": ("vocab_size", "mel_bins", "width", "layers", "heads"),
    "vocoder": ("mel_bins", "channels", "hop_length"),
    "3d": ("vocab_size", "width", "points", "layers", "heads", "diffusion_steps"),
    "research": ("vocab_size", "max_length", "width", "layers", "heads", "embedding_dim"),
    "document": ("vocab_size", "layout_vocab_size", "max_length", "width", "layers", "tag_classes"),
    "safety": ("vocab_size", "max_length", "width", "layers", "reason_classes", "decisions"),
}
SPECIALIST_INTEGER_CONFIG_KEYS = {
    "vocab_size", "block_size", "n_layer", "n_head", "n_embd", "image_size",
    "patch_size", "max_text_length", "width", "layers", "heads",
    "projection_dim", "text_length", "time_dim", "diffusion_steps",
    "n_mels", "blank_id", "mel_bins", "points", "max_length",
    "embedding_dim", "layout_vocab_size", "tag_classes", "channels",
    "hop_length", "reason_classes", "decisions",
}


def _validate_specialist_config(model_id: str, config: dict[str, Any]) -> None:
    """Catch malformed training and architecture settings before a model is queued."""
    required_training = {
        "seed", "max_steps", "batch_size", "learning_rate", "weight_decay",
        "grad_clip", "eval_interval", "save_interval",
    }
    missing_training = required_training - set(config)
    if missing_training:
        raise FamilyTrainingError(
            "Specialist training config is missing: " + ", ".join(sorted(missing_training))
        )
    for key in ("seed", "max_steps", "batch_size", "eval_interval", "save_interval"):
        require_int(config[key], key, minimum=0 if key == "seed" else 1)
    require_finite_number(config["learning_rate"], "learning_rate", minimum=0, minimum_inclusive=False)
    require_finite_number(config["weight_decay"], "weight_decay", minimum=0)
    require_finite_number(config["grad_clip"], "grad_clip", minimum=0, minimum_inclusive=False)
    model_config = config.get("model_config")
    if not isinstance(model_config, dict) or not model_config:
        raise FamilyTrainingError("A non-empty model_config object is required.")
    validate_numeric_tree(model_config)
    missing_model = set(SPECIALIST_MODEL_CONFIG_KEYS[model_id]) - set(model_config)
    if missing_model:
        raise FamilyTrainingError(
            "Specialist model_config is missing: " + ", ".join(sorted(missing_model))
        )
    for key in SPECIALIST_INTEGER_CONFIG_KEYS & set(model_config):
        require_int(
            model_config[key], f"model_config.{key}",
            minimum=0 if key == "blank_id" else 1,
        )


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
        _validate_specialist_config(model_id, config)
        output_dir = _resolve_output_dir(repo, config.get("output_dir"))
        manifest_path = repo / "data" / "specialists" / "manifests" / f"{model_id}.json"
        from aethrion.specialist_data import validate_specialist_dataset
        approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
        tokenizer_vocab_size = approval.get("tokenizer_vocab_size")
        configured_vocab_size = config["model_config"].get("vocab_size")
        if tokenizer_vocab_size is not None and (
            type(configured_vocab_size) is not int or configured_vocab_size < tokenizer_vocab_size
        ):
            raise FamilyTrainingError(
                "Model vocab_size is smaller than the verified dataset tokenizer vocabulary."
            )
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


_STOP_REQUESTED = False
_ACTIVE_CHILD: subprocess.Popen | None = None


def _request_family_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except (OSError, ProcessLookupError):
            pass


def _append_session_event(path: Path, event: dict[str, Any]) -> None:
    """Append one durable, newline-delimited family-training event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_family(
    root: str | Path = ".",
    *,
    selected: list[str] | None = None,
    execute: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Preflight or sequentially execute approved models with durable session journaling."""
    global _STOP_REQUESTED, _ACTIVE_CHILD
    repo = Path(root).resolve()
    model_ids = list(MODEL_IDS) if selected is None else list(selected)
    if not model_ids:
        raise FamilyTrainingError("At least one model ID must be selected.")
    if len(set(model_ids)) != len(model_ids):
        raise FamilyTrainingError("Duplicate model IDs are not allowed.")
    unknown = sorted(set(model_ids) - set(MODEL_IDS))
    if unknown:
        raise FamilyTrainingError("Unknown model IDs: " + ", ".join(unknown))

    session_id = uuid.uuid4().hex
    session_dir = repo / "artifacts" / "training_sessions"
    progress_path = session_dir / f"{session_id}.json"
    events_path = session_dir / f"{session_id}.jsonl"
    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    _STOP_REQUESTED = False
    previous_handlers = {}
    if execute:
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[stop_signal] = signal.signal(stop_signal, _request_family_stop)

    def persist(status: str, current_model: str | None = None) -> None:
        if not execute:
            return
        snapshot = {
            "format": "aethrion-family-training-session-v1",
            "session_id": session_id,
            "status": status,
            "current_model": current_model,
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execute_requested": True,
            "model_count": len(model_ids),
            "completed_count": sum(row.get("status") not in {"running", "blocked"} for row in rows),
            "blocked_count": sum(row.get("status") == "blocked" for row in rows),
            "models": rows,
            "released_count": 0,
        }
        write_json_atomic(snapshot, progress_path)

    try:
        if execute:
            _append_session_event(events_path, {
                "event": "session_started", "session_id": session_id,
                "timestamp_utc": started_at, "models": model_ids,
            })
            persist("running")

        for model_id in model_ids:
            if _STOP_REQUESTED:
                rows.append({"model_id": model_id, "ready": False, "status": "not_started_interrupted"})
                break
            try:
                plan = preflight_model(repo, model_id)
                command = list(plan["command"])
                if python_executable:
                    command[0] = python_executable
                if not execute:
                    rows.append({**plan, "status": "ready_not_started"})
                    continue

                rows.append({**plan, "status": "running"})
                persist("running", model_id)
                _append_session_event(events_path, {
                    "event": "model_started", "session_id": session_id,
                    "model_id": model_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": plan["checkpoint"],
                })
                try:
                    _ACTIVE_CHILD = subprocess.Popen(command, cwd=repo)
                    returncode = int(_ACTIVE_CHILD.wait())
                except KeyboardInterrupt:
                    _STOP_REQUESTED = True
                    if _ACTIVE_CHILD is not None and _ACTIVE_CHILD.poll() is None:
                        try:
                            _ACTIVE_CHILD.send_signal(signal.SIGINT)
                        except (OSError, ProcessLookupError):
                            pass
                    returncode = int(_ACTIVE_CHILD.wait()) if _ACTIVE_CHILD is not None else 130
                finally:
                    _ACTIVE_CHILD = None
                row = rows[-1]
                row["returncode"] = returncode
                row["status"] = (
                    "training_interrupted_not_released"
                    if _STOP_REQUESTED
                    else _training_outcome(repo / plan["output_dir"], model_id, returncode)
                )
                row.pop("ready", None)
                _append_session_event(events_path, {
                    "event": "model_finished", "session_id": session_id,
                    "model_id": model_id, "status": row["status"],
                    "returncode": returncode,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                persist("interrupted" if _STOP_REQUESTED else "running", model_id if _STOP_REQUESTED else None)
                if _STOP_REQUESTED:
                    break
            except Exception as exc:
                row = {
                    "model_id": model_id, "ready": False, "status": "blocked",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                rows.append(row)
                if execute:
                    _append_session_event(events_path, {
                        "event": "model_blocked", "session_id": session_id,
                        "model_id": model_id, "reason": row["reason"],
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    persist("running")

        interrupted = _STOP_REQUESTED or any(
            row.get("status") == "not_started_interrupted" for row in rows
        )
        report = {
            "format": "aethrion-family-training-run-v1",
            "session_id": session_id,
            "execution_policy": "local_only",
            "model_count": len(rows),
            "ready_count": sum(row.get("ready") is True for row in rows),
            "blocked_count": sum(row.get("status") == "blocked" for row in rows),
            "training_started": bool(execute and any(
                row.get("status") in {
                    "running", "training_completed_not_released", "training_failed_not_released",
                    "training_interrupted_not_released", "trainer_exited_without_verified_report",
                    "training_report_invalid_not_released", "training_progress_invalid_not_released",
                    "trainer_report_unrecognized_not_released",
                    "instruction_tuning_completed_not_released",
                } for row in rows
            )),
            "released_count": 0,
            "execute_requested": bool(execute),
            "interrupted": bool(interrupted),
            "models": rows,
            "note": (
                "A successful training process is not a release. Independent held-out "
                "evaluation, safety review, artifact integrity and human release approval remain mandatory."
            ),
        }
        if execute:
            final_status = "interrupted" if interrupted else "completed"
            persist(final_status)
            _append_session_event(events_path, {
                "event": "session_" + final_status, "session_id": session_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "blocked_count": report["blocked_count"],
            })
            write_json_atomic({**report, "status": final_status}, progress_path)
        return report
    finally:
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)


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
        write_json_atomic(report, target)
    return 0 if report["blocked_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
