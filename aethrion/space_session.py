"""One bounded, resumable training session: pull -> train (gate-checked) -> push.

Training itself is delegated to ``python -m aethrion.train_family --execute``, which
fails closed while data/governance gates are closed; this wrapper never opens a gate,
never trains more than one model per call and never marks a model trained/released.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

from aethrion import artifact_sync
from aethrion.train_family import MODEL_IDS, _resolve_output_dir

MAX_SESSION_STEPS = 500


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


def _stage_specialist_tokenizer(repo_root: Path, model: str, output_dir: Path) -> list[str]:
    """Stage only the tokenizer already approved and SHA-verified by the dataset manifest."""
    if model == "vocoder":
        return []
    from aethrion.specialist_data import validate_specialist_dataset, sha256_file

    manifest = repo_root / "data" / "specialists" / "manifests" / f"{model}.json"
    approval = validate_specialist_dataset(manifest, require_training_approval=True)
    source_value = approval.get("tokenizer_path")
    expected_hash = approval.get("tokenizer_sha256")
    if not isinstance(source_value, str) or not source_value:
        raise artifact_sync.ArtifactSyncError("Validated specialist manifest did not return a tokenizer path.")
    source = Path(source_value).resolve()
    try:
        source.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise artifact_sync.ArtifactSyncError("Validated tokenizer path escapes the repository root.") from exc
    if not source.is_file() or not isinstance(expected_hash, str) or sha256_file(source) != expected_hash.lower():
        raise artifact_sync.ArtifactSyncError("Specialist tokenizer is missing or its approved SHA-256 changed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "tokenizer.json"
    target_tmp = output_dir / "tokenizer.json.part"
    sidecar = output_dir / "tokenizer.json.sha256"
    sidecar_tmp = output_dir / "tokenizer.json.sha256.part"
    try:
        import shutil
        shutil.copyfile(source, target_tmp)
        if sha256_file(target_tmp) != expected_hash.lower():
            raise artifact_sync.ArtifactSyncError("Staged specialist tokenizer failed SHA-256 verification.")
        sidecar_tmp.write_text(expected_hash.lower() + "\n", encoding="ascii")
        target_tmp.replace(target)
        sidecar_tmp.replace(sidecar)
    finally:
        target_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
    return ["tokenizer.json", "tokenizer.json.sha256"]


def run_session(model: str, steps: int, *, root: str | Path = ".", api: Any = None, environ=None,
                runner: Callable[..., Any] | None = None, timeout: int = 3300,
                prepare_chat: bool = False) -> dict[str, Any]:
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
    pulled = artifact_sync.pull_model_artifacts(model, directory, api=api, environ=environ)
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
    result = (runner or subprocess.run)(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout)
    ok = getattr(result, "returncode", 1) == 0
    if ok and directory.is_dir() and model != "vocoder":
        _stage_specialist_tokenizer(repo_root, model, directory)
    pushed = artifact_sync.push_model_artifacts(model, directory, api=api, environ=environ) if ok and directory.is_dir() else []
    return {"model": model, "steps": steps, "pulled": pulled, "pushed": pushed, "trainer_ok": ok,
            "released": False, "progress": read_progress(directory) if directory.is_dir() else {"status": "unknown", "step": None},
            "note": "A session is not a release; evaluation and human approval are still required.",
            "trainer_tail": (getattr(result, "stdout", "") or "")[-400:] if ok else (getattr(result, "stderr", "") or "")[-400:]}
