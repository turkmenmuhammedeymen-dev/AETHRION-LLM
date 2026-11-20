"""Private Hugging Face Dataset-backed checkpoint persistence for Space sessions.

This stores only training artifacts. It never creates a repository, makes one public,
or logs credentials. A private Dataset repo and HF_TOKEN must be configured explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

MODEL_IDS = frozenset({
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety", "vocoder",
})
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ArtifactStoreError(RuntimeError):
    """A remote checkpoint cannot be safely stored or restored."""


def _configuration(environ: dict[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    token = str(env.get("HF_TOKEN", "")).strip()
    repo_id = str(env.get("AETHRION_CHECKPOINT_REPO", "")).strip()
    if not token or not repo_id:
        raise ArtifactStoreError(
            "Private checkpoint storage is not configured. Set HF_TOKEN and "
            "AETHRION_CHECKPOINT_REPO as Space Secrets."
        )
    if "/" not in repo_id or len(repo_id) > 96 or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ArtifactStoreError("AETHRION_CHECKPOINT_REPO must be an owner/repository ID.")
    return token, repo_id


def _model_id(value: str) -> str:
    if not isinstance(value, str) or value not in MODEL_IDS or not _MODEL_ID.fullmatch(value):
        raise ArtifactStoreError("Unknown or unsafe AETHRION model ID.")
    return value


def _private_dataset_api(token: str, repo_id: str):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise ArtifactStoreError(
            "Private checkpoint repository could not be accessed; no local-only fallback is allowed."
        ) from exc
    if getattr(info, "private", None) is not True:
        raise ArtifactStoreError("Checkpoint repository must already exist and be private.")
    return api


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checkpoint(model_id: str, checkpoint_path: str | Path, *, environ=None) -> dict[str, Any]:
    """Upload an immutable, checksum-addressed checkpoint then atomically publish its pointer.

    One writer per model is supported. The latest pointer is uploaded last, so an
    interrupted upload cannot advertise an incomplete checkpoint generation.
    """
    model = _model_id(model_id)
    source = Path(checkpoint_path)
    if not source.is_file() or source.is_symlink():
        raise ArtifactStoreError("Checkpoint must be an existing regular, non-symlink file.")
    token, repo_id = _configuration(environ)
    api = _private_dataset_api(token, repo_id)
    digest = _digest(source)
    filename = f"checkpoint-{digest}.pt"
    sidecar = f"checkpoint-{digest}.sha256"
    prefix = f"checkpoints/{model}"
    try:
        api.upload_file(
            path_or_fileobj=str(source), path_in_repo=f"{prefix}/{filename}",
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"AETHRION {model}: upload verified checkpoint generation",
        )
        with tempfile.TemporaryDirectory(prefix="aethrion-checkpoint-") as tmp:
            checksum_path = Path(tmp) / sidecar
            checksum_path.write_text(digest + "\n", encoding="ascii")
            api.upload_file(
                path_or_fileobj=str(checksum_path), path_in_repo=f"{prefix}/{sidecar}",
                repo_id=repo_id, repo_type="dataset",
                commit_message=f"AETHRION {model}: publish checkpoint checksum",
            )
            pointer = {
                "format": "aethrion-private-checkpoint-pointer-v1",
                "model_id": model,
                "sha256": digest,
                "checkpoint": filename,
                "checksum": sidecar,
            }
            pointer_path = Path(tmp) / "latest.json"
            pointer_path.write_text(
                json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8"
            )
            api.upload_file(
                path_or_fileobj=str(pointer_path), path_in_repo=f"{prefix}/latest.json",
                repo_id=repo_id, repo_type="dataset",
                commit_message=f"AETHRION {model}: publish verified checkpoint pointer",
            )
    except Exception as exc:
        raise ArtifactStoreError(
            "Checkpoint upload failed before a verified latest pointer was published."
        ) from exc
    return {"model_id": model, "sha256": digest, "status": "stored_private"}


def restore_checkpoint(model_id: str, destination: str | Path, *, environ=None) -> dict[str, Any]:
    """Restore only the immutable generation named by a validated private pointer."""
    model = _model_id(model_id)
    token, repo_id = _configuration(environ)
    _private_dataset_api(token, repo_id)
    try:
        from huggingface_hub import hf_hub_download
        pointer_file = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"checkpoints/{model}/latest.json", token=token,
        )
        pointer = json.loads(Path(pointer_file).read_text(encoding="utf-8"))
        if (
            not isinstance(pointer, dict)
            or pointer.get("format") != "aethrion-private-checkpoint-pointer-v1"
            or pointer.get("model_id") != model
            or not isinstance(pointer.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", pointer["sha256"])
            or pointer.get("checkpoint") != f"checkpoint-{pointer['sha256']}.pt"
            or pointer.get("checksum") != f"checkpoint-{pointer['sha256']}.sha256"
        ):
            raise ArtifactStoreError("Remote checkpoint pointer is invalid.")
        checkpoint_file = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"checkpoints/{model}/{pointer['checkpoint']}", token=token,
        )
        checksum_file = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"checkpoints/{model}/{pointer['checksum']}", token=token,
        )
        expected = Path(checksum_file).read_text(encoding="ascii").strip()
        source = Path(checkpoint_file)
        actual = _digest(source)
        if expected != pointer["sha256"] or actual != pointer["sha256"]:
            raise ArtifactStoreError("Remote checkpoint SHA-256 verification failed.")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download.tmp")
        sidecar = target.with_name(target.name + ".sha256")
        sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
        try:
            # Stream large model files instead of materializing the entire checkpoint in RAM.
            shutil.copyfile(source, temporary)
            if _digest(temporary) != actual:
                raise ArtifactStoreError("Restored checkpoint failed local read-back verification.")
            sidecar_tmp.write_text(actual + "\n", encoding="ascii")
            os.replace(temporary, target)
            os.replace(sidecar_tmp, sidecar)
        finally:
            temporary.unlink(missing_ok=True)
            sidecar_tmp.unlink(missing_ok=True)
        return {"model_id": model, "sha256": actual, "status": "restored_verified"}
    except ArtifactStoreError:
        raise
    except Exception as exc:
        raise ArtifactStoreError(
            "Private checkpoint restore failed; training must not resume from an unverified file."
        ) from exc
