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

ALLOWED_MODELS = frozenset({"chat", "code", "vision", "image", "asr", "tts", "vocoder", "3d", "research", "document", "safety", "base", "chat_instruct"})
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
# Keep the complete private, resumable inference package together. Checkpoints
# without their tokenizer/processor and provenance cannot be loaded in a fresh
# Space session. Reports and telemetry are retained privately for audit/resume;
# none of these files is a public release or changes a model's approval status.
ALLOWED_ARTIFACT_NAMES = frozenset({
    "checkpoint_last.pt",
    "checkpoint_last.pt.sha256",
    "checkpoint_last.pt.previous",
    "checkpoint_last.pt.previous.sha256",
    "tokenizer.json",
    "tokenizer.json.provenance.json",
    "processor.json",
    "training_report.json",
    "progress.json",
    "metrics.jsonl",
})


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


def _require_private(api: Any, repo: str, *, create: bool = False) -> None:
    """Refuse to touch a public artifact repo. Only a push may create it; a pull needs read access only."""
    try:
        if create:
            api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo, repo_type="model")
    except ArtifactSyncError:
        raise
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise ArtifactSyncError(
                f"Hugging Face rejected the token (HTTP {status}). Set a valid HF_TOKEN secret "
                f"({'with write access' if create else 'with read access'}) that can reach {repo}.") from exc
        raise
    if not getattr(info, "private", False):
        raise ArtifactSyncError("Artifact repo is not private; refusing to sync training artifacts.")


def _safe(model: str, name: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ArtifactSyncError(f"Unknown model '{model}'.")
    if not _NAME.fullmatch(name) or name not in ALLOWED_ARTIFACT_NAMES:
        raise ArtifactSyncError(f"File name not allowed: {name!r}")


def _check_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ArtifactSyncError(f"Unknown model '{model}'.")


def push_model_artifacts(model: str, directory: str | Path, *, api: Any = None, environ=None) -> list[str]:
    _check_model(model)
    repo = _repo_id(environ); client = _client(api, environ); _require_private(client, repo, create=True)
    directory = Path(directory)
    if not directory.is_dir():
        raise ArtifactSyncError("Artifact directory does not exist.")
    # Validate the complete upload set before the first network write. Keep
    # tokenizer/processor provenance and training telemetry with the checkpoint
    # in the private repo so a fresh Space session can resume and be audited.
    candidates = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        _safe(model, path.name)
        candidates.append(path)
    sent = []
    if candidates and hasattr(client, "create_commit"):
        # One commit per push: the Hub limits repository commits (128/hour), so per-file commits exhaust it.
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=f"{model}/{path.name}", path_or_fileobj=str(path)) for path in candidates]
        client.create_commit(repo_id=repo, repo_type="model", operations=ops,
                             commit_message=f"{model}: verified training artifacts (not a release)")
        return [path.name for path in candidates]
    for path in candidates:
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
