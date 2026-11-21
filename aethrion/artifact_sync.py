"""Private artifact sync so resumable training survives ZeroGPU request boundaries.

Checkpoints/sidecars are mirrored to a PRIVATE Hugging Face model repo named by
AETHRION_ARTIFACT_REPO (token from the Space secret HF_TOKEN). The sync refuses
to run if the repo is public, never creates a public repo, never logs the token,
and only moves allow-listed file names under artifacts/models/<model>/.
Integrity is still verified by aethrion.checkpoint after pull; this module only moves bytes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ALLOWED_MODELS = frozenset({"chat", "code", "vision", "image", "asr", "tts", "vocoder", "3d", "research", "document", "safety", "base"})
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
ALLOWED_SUFFIXES = (".pt", ".json", ".sha256", ".npz")


class ArtifactSyncError(RuntimeError):
    pass


def _repo_id(environ=None) -> str:
    env = os.environ if environ is None else environ
    repo = str(env.get("AETHRION_ARTIFACT_REPO", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
        raise ArtifactSyncError("AETHRION_ARTIFACT_REPO must be set to a private 'owner/name' repo.")
    return repo


def _client(api: Any = None, environ=None):
    if api is not None:
        return api
    env = os.environ if environ is None else environ
    token = env.get("HF_TOKEN")
    if not token:
        raise ArtifactSyncError("HF_TOKEN is not configured as a Space secret.")
    from huggingface_hub import HfApi  # imported lazily: not needed for tests
    return HfApi(token=token)


def _require_private(api: Any, repo: str) -> None:
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo, repo_type="model")
    if not getattr(info, "private", False):
        raise ArtifactSyncError("Artifact repo is not private; refusing to sync training artifacts.")


def _safe(model: str, name: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ArtifactSyncError(f"Unknown model '{model}'.")
    if not _NAME.fullmatch(name) or not name.endswith(ALLOWED_SUFFIXES) or name.startswith("."):
        raise ArtifactSyncError(f"File name not allowed: {name!r}")


def _check_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ArtifactSyncError(f"Unknown model '{model}'.")


def push_model_artifacts(model: str, directory: str | Path, *, api: Any = None, environ=None) -> list[str]:
    _check_model(model)
    repo = _repo_id(environ); client = _client(api, environ); _require_private(client, repo)
    directory = Path(directory); sent = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        _safe(model, path.name)
        client.upload_file(path_or_fileobj=str(path), path_in_repo=f"{model}/{path.name}", repo_id=repo, repo_type="model")
        sent.append(path.name)
    return sent


def pull_model_artifacts(model: str, directory: str | Path, *, api: Any = None, environ=None) -> list[str]:
    _check_model(model)
    repo = _repo_id(environ); client = _client(api, environ); _require_private(client, repo)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); got = []
    for remote in client.list_repo_files(repo_id=repo, repo_type="model"):
        prefix, _, name = remote.partition("/")
        if prefix != model or "/" in name:
            continue
        _safe(model, name)
        data = client.hf_hub_download(repo_id=repo, repo_type="model", filename=remote, local_dir=str(directory.parent / ".pull"))
        target = directory / name
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(Path(data).read_bytes())
        os.replace(tmp, target)
        got.append(name)
    return got
