"""One bounded, resumable training session: pull -> train (gate-checked) -> push.

Training itself is delegated to ``python -m aethrion.train_family --execute``, which
fails closed while data/governance gates are closed; this wrapper never opens a gate,
never trains more than one model per call and never marks a model trained/released.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from aethrion import artifact_sync
from aethrion.train_family import _resolve_output_dir

MAX_SESSION_STEPS = 500


def run_session(model: str, steps: int, *, root: str | Path = ".", api: Any = None, environ=None,
                runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    if model not in artifact_sync.ALLOWED_MODELS:
        raise artifact_sync.ArtifactSyncError(f"Unknown model '{model}'.")
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
    try:
        directory = _resolve_output_dir(repo_root, config.get("output_dir"))
    except ValueError as exc:
        raise artifact_sync.ArtifactSyncError(f"Unsafe training output directory: {exc}") from exc
    pulled = artifact_sync.pull_model_artifacts(model, directory, api=api, environ=environ)
    cmd = [sys.executable, "-m", "aethrion.train_family", "--execute", "--models", model, "--session-steps", str(steps)]
    result = (runner or subprocess.run)(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=3300)
    ok = getattr(result, "returncode", 1) == 0
    pushed = artifact_sync.push_model_artifacts(model, directory, api=api, environ=environ) if ok and directory.is_dir() else []
    return {"model": model, "steps": steps, "pulled": pulled, "pushed": pushed, "trainer_ok": ok,
            "released": False, "note": "A session is not a release; evaluation and human approval are still required.",
            "trainer_tail": (getattr(result, "stdout", "") or "")[-400:] if ok else (getattr(result, "stderr", "") or "")[-400:]}
