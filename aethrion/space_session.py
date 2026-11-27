"""One bounded, resumable training session: pull -> train (gate-checked) -> push.

Training itself is delegated to ``python -m aethrion.train_family --execute``, which
fails closed while data/governance gates are closed; this wrapper never opens a gate,
never trains more than one model per call and never marks a model trained/released.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

from aethrion import artifact_sync
from aethrion.train_family import MODEL_IDS, _resolve_output_dir

MAX_SESSION_STEPS = 500
# Trainer refusals that mean the pulled checkpoint belongs to an older dataset/config.
STALE_RESUME_MARKERS = (
    "Resume model or approved dataset differs from checkpoint.",
    "Resume model configuration differs from checkpoint.",
    "Specialist source/approval manifest changed; refusing unsafe resume.",
    "Specialist training settings changed; refusing a non-reproducible resume.",
    "BASE training settings changed; refusing a non-reproducible resume.",
)
CHECKPOINT_STATE_NAMES = (
    "checkpoint_last.pt", "checkpoint_last.pt.sha256",
    "checkpoint_last.pt.previous", "checkpoint_last.pt.previous.sha256",
    "training_report.json", "progress.json", "metrics.jsonl",
)


def _is_stale_resume(result: Any) -> bool:
    text = f"{getattr(result, 'stdout', '') or ''}\n{getattr(result, 'stderr', '') or ''}"
    return getattr(result, "returncode", 1) != 0 and any(marker in text for marker in STALE_RESUME_MARKERS)


def quarantine_stale_checkpoint(directory: Path, model: str) -> list[str]:
    """Move incompatible local resume state aside (never deleted) so a fresh run can start."""
    import time
    target = Path(directory).parent / ".stale" / f"{model}-{int(time.time())}"
    moved = []
    for name in CHECKPOINT_STATE_NAMES:
        source = Path(directory) / name
        if source.is_file():
            target.mkdir(parents=True, exist_ok=True)
            os.replace(source, target / name)
            moved.append(name)
    return moved


def graceful_runner(cmd, *, cwd=None, capture_output=True, text=True, timeout=85, grace=15, **_):
    """Run the trainer; at the deadline send SIGTERM so it pauses cleanly after a verified
    checkpoint (status ``training_interrupted_not_released``) instead of being killed."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return SimpleNamespace(returncode=124, stdout=out or "", stderr=(err or "") + "\nkilled after grace period")
    return SimpleNamespace(returncode=proc.returncode, stdout=out or "", stderr=err or "")


def read_progress(directory: Path) -> dict[str, Any]:
    """Summarise training_report.json (read-only). Never claims release."""
    try:
        report = json.loads((Path(directory) / "training_report.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # BASE (CHAT) trainer writes progress.json while a session is paused; the report appears only at the end.
        try:
            prog = json.loads((Path(directory) / "progress.json").read_text(encoding="utf-8"))
            step = prog.get("step")
            if prog.get("status") == "interrupted" and isinstance(step, int):
                return {"status": "training_interrupted_not_released", "step": step + 1,
                        "validation_loss": prog.get("val_loss", prog.get("validation_loss")),
                        "held_out_test_loss": None, "complete": False}
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            pass
        return {"status": "unknown", "step": None}
    if not isinstance(report, dict):
        return {"status": "unknown", "step": None}
    step = report.get("steps", report.get("step"))
    if report.get("status") == "training_in_progress_not_released" and isinstance(step, int):
        step += 1
    return {"status": str(report.get("status", "unknown")),
            "step": step if isinstance(step, int) else None,
            "validation_loss": report.get("validation_loss"),
            "held_out_test_loss": report.get("held_out_test_loss"),
            "complete": report.get("status") == "training_complete_not_released"}


def run_session(model: str, steps: int, *, root: str | Path = ".", api: Any = None, environ=None,
                runner: Callable[..., Any] | None = None, timeout: int = 3300,
                prepare_chat: bool = False, fresh_on_stale: bool = False) -> dict[str, Any]:
    # artifact_sync retains a legacy "base" identifier for artifact operations,
    # but train_family does not expose it as a trainable model. Reject it before
    # downloading anything so the session API and trainer share one model list.
    if model not in MODEL_IDS:
        raise artifact_sync.ArtifactSyncError(f"Unknown or unsupported training model '{model}'.")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_SESSION_STEPS:
        raise ValueError(f"steps must be an integer between 1 and {MAX_SESSION_STEPS}.")
    repo_root = Path(root).resolve()
    config_path = (
        repo_root / "configs" / "small.json"
        if model == "chat"
        else repo_root / "configs" / "specialists" / f"{model}.json"
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise artifact_sync.ArtifactSyncError(f"Training config is missing or invalid: {config_path}") from exc
    if not isinstance(config, dict):
        raise artifact_sync.ArtifactSyncError("Training config must be a JSON object.")
    if model != "chat" and config.get("model_id") != model:
        raise artifact_sync.ArtifactSyncError(
            f"Training config model_id does not match selected model '{model}'."
        )
    try:
        directory = _resolve_output_dir(repo_root, config.get("output_dir"))
    except ValueError as exc:
        raise artifact_sync.ArtifactSyncError(f"Unsafe training output directory: {exc}") from exc
    try:
        pulled = artifact_sync.pull_model_artifacts(model, directory, api=api, environ=environ)
    except Exception as exc:
        detail = f"Private artifact pull failed: {type(exc).__name__}: {str(exc)[:240]}"
        return {
            "model": model, "steps": steps, "pulled": [], "pushed": [],
            "trainer_ok": False, "released": False,
            "progress": {"status": "artifact_sync_pull_failed_not_released", "step": None, "complete": False},
            "note": "Training did not start because private artifact retrieval failed.",
            "trainer_tail": detail,
        }
    if model == "chat" and prepare_chat:
        # Deterministic, gate-checked BASE corpus + HMAC-signed audit + tokenizer (needs the Space secret key).
        prep = subprocess.run([sys.executable, "scripts/prepare_base.py"], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=300)
        if prep.returncode != 0:
            return {"model": model, "steps": steps, "pulled": pulled, "pushed": [], "trainer_ok": False, "released": False,
                    "progress": {"status": "base_preparation_failed", "step": None, "complete": False},
                    "note": "BASE corpus preparation failed; nothing was trained.",
                    "trainer_tail": (prep.stderr or prep.stdout or "")[-1500:]}
    cmd = [sys.executable, "-m", "aethrion.train_family", "--execute", "--models", model, "--session-steps", str(steps)]
    run = runner or subprocess.run
    result = run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout)
    stale_replaced: list[str] = []
    if _is_stale_resume(result):
        if not fresh_on_stale:
            return {"model": model, "steps": steps, "pulled": pulled, "pushed": [], "trainer_ok": False,
                    "released": False,
                    "progress": {"status": "stale_checkpoint_incompatible_not_released", "step": None, "complete": False},
                    "note": "The stored checkpoint was trained on a different dataset/config and cannot be resumed.",
                    "trainer_tail": (str(getattr(result, "stderr", "")) or str(getattr(result, "stdout", "")))[-1500:]}
        stale_replaced = quarantine_stale_checkpoint(directory, model)
        result = run(cmd[:], cwd=str(repo_root), capture_output=True, text=True, timeout=timeout)
    ok = getattr(result, "returncode", 1) == 0
    try:
        pushed = artifact_sync.push_model_artifacts(model, directory, api=api, environ=environ) if ok and directory.is_dir() else []
    except Exception as exc:
        detail = f"Private artifact push failed after trainer completion: {type(exc).__name__}: {str(exc)[:240]}"
        return {
            "model": model, "steps": steps, "pulled": pulled, "pushed": [],
            "trainer_ok": False, "released": False,
            "progress": {"status": "artifact_sync_push_failed_not_released", "step": None, "complete": False},
            "note": "Trainer output was not mirrored to the private artifact repository; no release was made.",
            "trainer_tail": detail,
        }
    progress = read_progress(directory) if directory.is_dir() else {"status": "unknown", "step": None, "complete": False}
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    diagnostics = "\n".join(part for part in (stdout, stderr) if part.strip())
    return {
        "model": model,
        "steps": steps,
        "pulled": pulled,
        "pushed": pushed,
        "trainer_ok": ok,
        "trainer_exit_code": int(getattr(result, "returncode", 1)),
        "stale_checkpoint_replaced": stale_replaced,
        "released": False,
        "progress": progress,
        "note": "A session is not a release; evaluation and human approval are still required.",
        "trainer_tail": diagnostics[-1500:] if diagnostics else "Trainer produced no stdout or stderr.",
    }
