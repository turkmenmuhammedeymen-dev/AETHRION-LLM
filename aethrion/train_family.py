"""Fail-closed sequential training orchestrator for all eleven native models.

The default mode is preflight-only. Training starts only with --execute and
only after the model-specific data validator accepts the exact local dataset.
No third-party model weights or inference APIs are used.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import signal
import subprocess
import torch
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aethrion.checkpoint import write_json_atomic
from aethrion.session_journal import append_session_event, audit_session_journal, audit_session_artifacts
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

    # If only the previous generation survived (for example after a crash
    # between artifact replacement steps), still invoke the trainer's verified
    # loader. It tries checkpoint_last.pt first and then checkpoint_last.pt.previous.
    previous_checkpoint = Path(str(checkpoint) + ".previous")
    if checkpoint.is_file() or previous_checkpoint.is_file():
        command.extend(["--resume", str(checkpoint.relative_to(repo))])

    return {
        "model_id": model_id,
        "ready": True,
        "command": command,
        "checkpoint": str(checkpoint.relative_to(repo)),
        "output_dir": str(output_dir.relative_to(repo)),
    }



def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _verify_attempt_checkpoint_binding(
    output_dir: Path,
    model_id: str,
    session_id: str,
    attempt_id: str,
    journal_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cross-check report, exact checkpoint generation and journal attempt lineage."""
    report_path = output_dir / "training_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"verified": False, "reason": "report_missing_or_invalid"}
    if not isinstance(report, dict):
        return {"verified": False, "reason": "report_not_object"}
    if (report.get("model_id") != model_id or report.get("session_id") != session_id
            or report.get("attempt_id") != attempt_id):
        return {"verified": False, "reason": "report_lineage_mismatch"}
    generation = report.get("checkpoint_generation")
    created_at = report.get("report_created_at_ns")
    digest = report.get("checkpoint_sha256")
    if type(generation) is not int or generation < 1 or type(created_at) is not int:
        return {"verified": False, "reason": "report_generation_or_timestamp_invalid"}
    if not isinstance(digest, str) or len(digest) != 64:
        return {"verified": False, "reason": "report_checkpoint_hash_invalid"}
    checkpoint = output_dir / "checkpoint_last.pt"
    candidates = [checkpoint, Path(str(checkpoint) + ".previous")]
    selected = next((candidate for candidate in candidates if candidate.is_file()
                     and hmac.compare_digest(digest.lower(), _file_sha256(candidate))), None)
    if selected is None:
        return {"verified": False, "reason": "no_checkpoint_generation_matches_report_hash"}
    try:
        if model_id == "chat":
            from aethrion.checkpoint import _load_verified_checkpoint
            state = _load_verified_checkpoint(selected, "cpu")
            checkpoint_step = state.get("step")
            report_step = report.get("step")
            if type(checkpoint_step) is not int or checkpoint_step != report_step:
                return {"verified": False, "reason": "checkpoint_report_step_mismatch"}
        else:
            from aethrion.specialist_train import _load_checkpoint
            state = _load_checkpoint(selected, torch.device("cpu"))
            checkpoint_step = state.get("step")
            report_steps = report.get("steps")
            if type(checkpoint_step) is not int or checkpoint_step + 1 != report_steps:
                return {"verified": False, "reason": "checkpoint_report_step_mismatch"}
            if state.get("model_id") != model_id:
                return {"verified": False, "reason": "checkpoint_model_mismatch"}
            for key in ("dataset_sha256", "tokenizer_sha256", "manifest_sha256"):
                if state.get(key) != report.get(key):
                    return {"verified": False, "reason": "checkpoint_report_provenance_mismatch"}
        if (state.get("session_id") != session_id or state.get("attempt_id") != attempt_id
                or state.get("checkpoint_generation") != generation):
            return {"verified": False, "reason": "checkpoint_lineage_or_generation_mismatch"}
        if model_id == "chat":
            if (state.get("data_sha256") != report.get("dataset_sha256")
                    or state.get("tokenizer_sha256") != report.get("tokenizer_sha256")):
                return {"verified": False, "reason": "checkpoint_report_provenance_mismatch"}
    except Exception:
        return {"verified": False, "reason": "checkpoint_semantic_validation_failed"}
    if journal_attempt is not None and journal_attempt.get("state") == "terminal_recorded":
        if (journal_attempt.get("checkpoint_generation") != generation
                or journal_attempt.get("checkpoint_sha256") != digest
                or journal_attempt.get("report_created_at_ns") != created_at):
            return {"verified": False, "reason": "journal_report_checkpoint_mismatch"}
    return {
        "verified": True,
        "checkpoint_generation": generation,
        "checkpoint_sha256": digest,
        "report_created_at_ns": created_at,
        "checkpoint_source": "latest" if selected == checkpoint else "previous",
        "checkpoint_step": checkpoint_step,
    }


def _finite_metric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _training_outcome(
    output_dir: Path,
    model_id: str,
    returncode: int,
    *,
    checkpoint_path: Path | None = None,
    started_at_ns: int | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    allow_interrupted: bool = False,
) -> str:
    """Accept only a fresh, model-matched report backed by the exact checkpoint bytes."""
    if returncode != 0 and not allow_interrupted:
        return "training_failed_not_released"
    report_path = output_dir / "training_report.json"
    if not report_path.is_file():
        return "trainer_exited_without_verified_report"
    try:
        if started_at_ns is not None and report_path.stat().st_mtime_ns < started_at_ns:
            return "training_report_stale_not_released"
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "training_report_invalid_not_released"
    if not isinstance(report, dict) or report.get("model_id") != model_id:
        return "training_report_model_mismatch_not_released"
    if report.get("release_status") != "NOT_RELEASED":
        return "training_report_release_state_invalid_not_released"
    if session_id is not None and report.get("session_id") != session_id:
        return "training_report_session_mismatch_not_released"
    if attempt_id is not None and report.get("attempt_id") != attempt_id:
        return "training_report_attempt_mismatch_not_released"
    report_created_at_ns = report.get("report_created_at_ns")
    if (
        type(report_created_at_ns) is not int
        or (started_at_ns is not None and report_created_at_ns < started_at_ns)
    ):
        return "training_report_timestamp_invalid_not_released"
    status = report.get("status")
    if status not in {"training_complete_not_released", "training_interrupted_not_released"}:
        return "trainer_report_unrecognized_not_released"
    if status == "training_complete_not_released" and returncode != 0:
        return "training_failed_not_released"

    checkpoint_path = checkpoint_path or (output_dir / "checkpoint_last.pt")
    try:
        expected_hash = report.get("checkpoint_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return "training_report_checkpoint_hash_invalid_not_released"
        # A crash may damage the latest generation while preserving .previous.
        # Bind the report to the exact generation hash it names.
        candidates = [checkpoint_path, Path(str(checkpoint_path) + ".previous")]
        matching = [candidate for candidate in candidates if candidate.is_file()
                    and hmac.compare_digest(expected_hash.lower(), _file_sha256(candidate))]
        if not matching:
            return "training_checkpoint_hash_mismatch_not_released"
        checkpoint_path = matching[0]
    except OSError:
        return "training_checkpoint_unreadable_not_released"

    interrupted = status == "training_interrupted_not_released"
    if model_id == "chat":
        metrics = report.get("metrics")
        valid = (
            type(report.get("step")) is int and report["step"] >= 0
            and isinstance(metrics, dict)
            and _finite_metric(metrics.get("train"))
            and _finite_metric(metrics.get("val"))
            and (interrupted or _finite_metric(report.get("held_out_test_loss")))
            and all(
                isinstance(report.get(key), str) and len(report[key]) == 64
                for key in ("dataset_sha256", "tokenizer_sha256")
            )
        )
    else:
        valid = (
            type(report.get("steps")) is int and report["steps"] >= 1
            and _finite_metric(report.get("validation_loss"))
            and (interrupted or _finite_metric(report.get("held_out_test_loss")))
            and all(
                isinstance(report.get(key), str) and len(report[key]) == 64
                for key in ("dataset_sha256", "tokenizer_sha256", "manifest_sha256")
            )
        )
    if not valid:
        return "training_report_metrics_invalid_not_released"

    try:
        if model_id == "chat":
            from aethrion.checkpoint import _load_verified_checkpoint
            checkpoint_state = _load_verified_checkpoint(checkpoint_path, map_location="cpu")
            checkpoint_step = checkpoint_state.get("step")
            if (
                type(report.get("checkpoint_generation")) is not int
                or report["checkpoint_generation"] < 1
                or checkpoint_state.get("checkpoint_generation") != report.get("checkpoint_generation")
                or checkpoint_state.get("session_id") != session_id
                or checkpoint_state.get("attempt_id") != attempt_id
            ):
                return "training_checkpoint_session_generation_mismatch_not_released"
            report_step = report["step"]
            if type(checkpoint_step) is not int or checkpoint_step != report_step:
                return "training_report_checkpoint_step_mismatch_not_released"
        else:
            from aethrion.specialist_train import _load_checkpoint
            checkpoint_state = _load_checkpoint(checkpoint_path, torch.device("cpu"))
            checkpoint_step = checkpoint_state.get("step")
            if (
                checkpoint_state.get("session_id") != session_id
                or checkpoint_state.get("attempt_id") != attempt_id
                or checkpoint_state.get("checkpoint_generation") != report.get("checkpoint_generation")
            ):
                return "training_checkpoint_session_generation_mismatch_not_released"
            report_steps = report["steps"]
            if (
                checkpoint_state.get("model_id") != model_id
                or type(checkpoint_step) is not int
                or checkpoint_step + 1 != report_steps
            ):
                return "training_report_checkpoint_step_mismatch_not_released"
            if checkpoint_state.get("dataset_sha256") != report.get("dataset_sha256"):
                return "training_report_checkpoint_provenance_mismatch_not_released"
            if checkpoint_state.get("tokenizer_sha256") != report.get("tokenizer_sha256"):
                return "training_report_checkpoint_provenance_mismatch_not_released"
            if checkpoint_state.get("manifest_sha256") != report.get("manifest_sha256"):
                return "training_report_checkpoint_provenance_mismatch_not_released"
        if model_id == "chat":
            if checkpoint_state.get("data_sha256") != report.get("dataset_sha256"):
                return "training_report_checkpoint_provenance_mismatch_not_released"
            if checkpoint_state.get("tokenizer_sha256") != report.get("tokenizer_sha256"):
                return "training_report_checkpoint_provenance_mismatch_not_released"
    except Exception:
        return "training_checkpoint_semantic_validation_failed_not_released"

    return "training_interrupted_not_released" if interrupted else "training_completed_not_released"


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
    """Append one durable, hash-chained family-training event."""
    append_session_event(path, event)


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
        journal_audit = audit_session_journal(events_path, session_id)
        if not journal_audit.get("valid"):
            raise FamilyTrainingError(
                "Session journal integrity failure: " + str(journal_audit.get("reason"))
            )
        snapshot = {
            "format": "aethrion-family-training-session-v2",
            "journal_integrity": "sha256_chain_verified",
            "journal_event_count": journal_audit["event_count"],
            "journal_last_event_sha256": journal_audit["last_event_sha256"],
            "session_id": session_id,
            "status": status,
            "current_model": current_model,
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execute_requested": True,
            "model_count": len(model_ids),
            "completed_count": sum(
                row.get("status") in {
                    "training_completed_not_released",
                    "instruction_tuning_completed_not_released",
                    "training_failed_not_released",
                    "training_interrupted_not_released",
                    "trainer_exited_without_verified_report",
                    "training_report_invalid_not_released",
                    "training_progress_invalid_not_released",
                    "trainer_report_unrecognized_not_released",
                }
                for row in rows
            ),
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

                attempt_id = uuid.uuid4().hex
                command.extend(["--session-id", session_id, "--attempt-id", attempt_id])
                rows.append({**plan, "status": "running", "process_started": False,
                             "attempt_id": attempt_id, "session_id": session_id})
                try:
                    report_started_at_ns = time.time_ns()
                    _append_session_event(events_path, {
                        "event": "model_attempt_started", "session_id": session_id,
                        "attempt_id": attempt_id, "model_id": model_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    persist("running", model_id)
                    _ACTIVE_CHILD = subprocess.Popen(command, cwd=repo)
                    rows[-1]["process_started"] = True
                    _append_session_event(events_path, {
                        "event": "model_started", "session_id": session_id,
                        "attempt_id": attempt_id, "model_id": model_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "checkpoint": plan["checkpoint"],
                    })
                    persist("running", model_id)
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
                row["status"] = _training_outcome(
                    repo / plan["output_dir"], model_id, returncode,
                    checkpoint_path=repo / plan["checkpoint"],
                    started_at_ns=report_started_at_ns,
                    session_id=session_id, attempt_id=attempt_id,
                    allow_interrupted=_STOP_REQUESTED,
                )
                row.pop("ready", None)
                verified_report = {}
                if row["status"] in {
                    "training_completed_not_released", "training_interrupted_not_released"
                }:
                    verified_report = json.loads(
                        (repo / plan["output_dir"] / "training_report.json").read_text(encoding="utf-8")
                    )
                _append_session_event(events_path, {
                    "event": "model_finished", "session_id": session_id,
                    "attempt_id": attempt_id,
                    "model_id": model_id, "status": row["status"],
                    "returncode": returncode,
                    "report_created_at_ns": verified_report.get("report_created_at_ns"),
                    "checkpoint_generation": verified_report.get("checkpoint_generation"),
                    "checkpoint_sha256": verified_report.get("checkpoint_sha256"),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                persist("interrupted" if _STOP_REQUESTED else "running", model_id if _STOP_REQUESTED else None)
                if _STOP_REQUESTED:
                    break
            except Exception as exc:
                if rows and rows[-1].get("model_id") == model_id and rows[-1].get("status") == "running":
                    row = rows[-1]
                    row["status"] = "blocked"
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                else:
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
                row.get("process_started") is True for row in rows
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
            _append_session_event(events_path, {
                "event": "session_" + final_status, "session_id": session_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "blocked_count": report["blocked_count"],
            })
            final_audit = audit_session_journal(events_path, session_id)
            if not final_audit.get("valid"):
                raise FamilyTrainingError(
                    "Final session journal integrity failure: " + str(final_audit.get("reason"))
                )
            report.update({
                "journal_integrity": "sha256_chain_verified",
                "journal_event_count": final_audit["event_count"],
                "journal_last_event_sha256": final_audit["last_event_sha256"],
            })
            persist(final_status)
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
    parser.add_argument(
        "--audit-session", default=None,
        help="Verify an existing session JSONL hash chain and matching snapshot; never starts training.",
    )
    args = parser.parse_args()
    if args.audit_session:
        repo_root = Path(args.root).resolve()
        result = audit_session_artifacts(
            repo_root / "artifacts" / "training_sessions",
            args.audit_session,
        )
        if result.get("valid"):
            attempts = result.get("attempts", {})
            reconstructed = result.get("reconstructed_session", {})
            for row in reconstructed.get("models", []):
                attempt_id = row.get("attempt_id")
                if not attempt_id:
                    continue
                attempt = attempts.get(attempt_id, {})
                output_dir = repo_root / row.get("output_dir", "")
                row["checkpoint_report_binding"] = _verify_attempt_checkpoint_binding(
                    output_dir, row["model_id"], args.audit_session, attempt_id, attempt,
                )
            reconstructed["automatic_resume"] = False
            result["checkpoint_cross_validation"] = "complete"
            result["valid"] = all(
                row.get("checkpoint_report_binding", {}).get("verified", False)
                for row in reconstructed.get("models", [])
                if row.get("attempt_id")
            )
            if not result["valid"]:
                result["recovery"] = "manual_review_required"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("valid") else 3
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
