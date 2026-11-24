"""Publish verified private AETHRION training checkpoints into the Space workspace.

This sync makes checkpoint files visible to the Space but deliberately does not
change models/manifest.json, mark models trained/released, or bypass evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

SPACE_ID = "vizyon2027/AETHRION-LLM"
ARTIFACT_REPO = "vizyon2027/aethrion-artifacts"
MODELS = (
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety", "vocoder",
)
SPECIALISTS_WITH_TOKENIZER = (
    "code", "vision", "image", "asr", "tts", "3d", "research", "document", "safety",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_download(api: HfApi, repo_id: str, filename: str, repo_type: str,
                      destination: Path, expected_sha: str | None = None,
                      revision: str | None = None) -> str:
    downloaded = Path(hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=filename,
        token=api.token, revision=revision,
    ))
    actual = sha256(downloaded)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(
            f"SHA-256 mismatch for {repo_id}/{filename}: expected {expected_sha}, got {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, destination)
    return actual


def read_sha_sidecar(api: HfApi, model: str, filename: str) -> str:
    remote = f"{model}/{filename}.sha256"
    local = Path(hf_hub_download(
        repo_id=ARTIFACT_REPO, repo_type="model", filename=remote, token=api.token,
    ))
    value = local.read_text(encoding="ascii").strip().split()[0].lower()
    if not SHA_RE.fullmatch(value):
        raise RuntimeError(f"Invalid SHA-256 sidecar: {remote}")
    return value


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN repository secret is missing.")
    api = HfApi(token=token)

    artifact_info = api.repo_info(repo_id=ARTIFACT_REPO, repo_type="model")
    if not getattr(artifact_info, "private", False):
        raise SystemExit("Refusing to read training checkpoints from a non-private artifact repository.")
    space_info = api.repo_info(repo_id=SPACE_ID, repo_type="space")
    artifact_files = set(api.list_repo_files(repo_id=ARTIFACT_REPO, repo_type="model"))
    space_revision = getattr(space_info, "sha", None)

    with tempfile.TemporaryDirectory(prefix="aethrion-space-artifacts-") as temp:
        staging = Path(temp)
        report: dict[str, object] = {
            "format": "aethrion-training-artifact-sync-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_artifact_repo": ARTIFACT_REPO,
            "source_artifact_revision": getattr(artifact_info, "sha", None),
            "target_space": SPACE_ID,
            "source_space_revision": space_revision,
            "policy": "checkpoint_available_not_released",
            "release_status_changed": False,
            "models": {},
        }
        model_rows: dict[str, object] = {}

        for model in MODELS:
            folder = staging / "artifacts" / "models" / model
            checkpoint = f"{model}/checkpoint_last.pt"
            if checkpoint not in artifact_files:
                raise RuntimeError(f"Missing required checkpoint: {checkpoint}")
            expected = read_sha_sidecar(api, model, "checkpoint_last.pt")
            actual = verified_download(
                api, ARTIFACT_REPO, checkpoint, "model",
                folder / "checkpoint_last.pt", expected,
            )
            (folder / "checkpoint_last.pt.sha256").write_text(actual + "\n", encoding="ascii")

            previous_name = "checkpoint_last.pt.previous"
            previous = f"{model}/{previous_name}"
            previous_status = "absent"
            if previous in artifact_files:
                previous_expected = read_sha_sidecar(api, model, previous_name)
                previous_actual = verified_download(
                    api, ARTIFACT_REPO, previous, "model",
                    folder / previous_name, previous_expected,
                )
                (folder / f"{previous_name}.sha256").write_text(
                    previous_actual + "\n", encoding="ascii"
                )
                previous_status = "verified"

            tokenizer_status = "not_applicable"
            tokenizer_sha = None
            if model in SPECIALISTS_WITH_TOKENIZER:
                manifest_name = f"data/specialists/manifests/{model}.json"
                manifest_file = Path(hf_hub_download(
                    repo_id=SPACE_ID, repo_type="space", filename=manifest_name,
                    token=token, revision=space_revision,
                ))
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                raw_path = manifest.get("tokenizer_path")
                expected_tokenizer = str(manifest.get("tokenizer_sha256", "")).lower()
                if not isinstance(raw_path, str) or not raw_path:
                    raise RuntimeError(f"{model}: tokenizer_path missing from source manifest.")
                relative = PurePosixPath(raw_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"{model}: unsafe tokenizer_path in source manifest.")
                source_tokenizer = str(PurePosixPath("data/specialists/manifests") / relative)
                if not SHA_RE.fullmatch(expected_tokenizer):
                    raise RuntimeError(f"{model}: invalid tokenizer_sha256 in source manifest.")
                tokenizer_sha = verified_download(
                    api, SPACE_ID, source_tokenizer, "space",
                    folder / "tokenizer.json", expected_tokenizer, revision=space_revision,
                )
                (folder / "tokenizer.json.sha256").write_text(
                    tokenizer_sha + "\n", encoding="ascii"
                )
                tokenizer_status = "verified_from_space_manifest"

            model_rows[model] = {
                "status": "checkpoint_available_not_released",
                "checkpoint_sha256": actual,
                "previous_checkpoint": previous_status,
                "tokenizer": tokenizer_status,
                "tokenizer_sha256": tokenizer_sha,
                "evaluation": "not_verified_by_this_sync",
                "release_approval": False,
            }

        # Chat tokenizer is optional here: it is not in the private artifact
        # repository and the current Space source does not advertise a chat
        # tokenizer manifest. Never synthesize one or guess its provenance.
        chat_tokenizer = "not_found"
        chat_path = "artifacts/tokenizer.json"
        if chat_path in set(api.list_repo_files(repo_id=SPACE_ID, repo_type="space")):
            chat_file = Path(hf_hub_download(
                repo_id=SPACE_ID, repo_type="space", filename=chat_path,
                token=token, revision=space_revision,
            ))
            digest = sha256(chat_file)
            target = staging / "artifacts" / "models" / "chat" / "tokenizer.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chat_file, target)
            (target.parent / "tokenizer.json.sha256").write_text(digest + "\n", encoding="ascii")
            chat_tokenizer = "copied_from_existing_space_file"
            model_rows["chat"]["tokenizer"] = chat_tokenizer
            model_rows["chat"]["tokenizer_sha256"] = digest

        report["models"] = model_rows
        manifest_path = staging / "models" / "training_artifact_sync.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        commit = api.upload_folder(
            repo_id=SPACE_ID,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Sync verified AETHRION training checkpoints from private artifact repository",
            allow_patterns=["artifacts/models/**", "models/training_artifact_sync.json"],
        )
        revision = getattr(commit, "oid", None)
        if not revision:
            raise RuntimeError("Space upload returned no verifiable commit SHA.")

        # Read back every checkpoint from the exact commit and verify byte integrity.
        for model in MODELS:
            filename = f"artifacts/models/{model}/checkpoint_last.pt"
            remote = Path(hf_hub_download(
                repo_id=SPACE_ID, repo_type="space", filename=filename,
                token=token, revision=revision,
            ))
            actual = sha256(remote)
            expected = model_rows[model]["checkpoint_sha256"]
            if actual != expected:
                raise RuntimeError(f"Space read-back SHA-256 mismatch: {filename}")

        current = api.repo_info(repo_id=SPACE_ID, repo_type="space")
        if getattr(current, "sha", None) != revision:
            raise RuntimeError("Space HEAD does not match the artifact-sync commit.")

        print(f"SPACE_ARTIFACT_SYNC_COMMIT={revision}")
        print(f"CHECKPOINTS_SYNCED={len(MODELS)}")
        print(f"CHAT_TOKENIZER={chat_tokenizer}")
        print("CHECKPOINT_READBACK_SHA256=PASS")
        print("MODEL_RELEASE_STATUS=UNCHANGED_NOT_RELEASED")


if __name__ == "__main__":
    main()
