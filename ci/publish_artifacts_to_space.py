"""Keep unapproved training checkpoints out of the Hugging Face Space.

Training checkpoints remain in the private aethrion-artifacts model repository.
The Space receives model files only through a separate, fully verified release
deployment. This cleanup prevents unapproved/partial checkpoints from consuming
the Space's 1 GB storage quota or appearing to be released models.
"""
from __future__ import annotations

import json
import os

from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

SPACE_ID = "vizyon2027/AETHRION-LLM"
MODELS = (
    "chat", "code", "vision", "image", "asr", "tts", "3d",
    "research", "document", "safety", "vocoder",
)


def release_approved_models(token: str) -> set[str]:
    manifest_path = hf_hub_download(
        repo_id=SPACE_ID, repo_type="space",
        filename="models/manifest.json", token=token,
    )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise RuntimeError("Space model manifest is missing or invalid; refusing cleanup.")
    approved = set()
    for model in MODELS:
        entry = payload["models"].get(model)
        if not isinstance(entry, dict) or entry.get("status") != "trained":
            continue
        if all(entry.get(flag) is True for flag in (
            "release_approval", "quality_gate_passed", "safety_gate_passed"
        )):
            approved.add(model)
    return approved


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN repository secret is missing.")
    api = HfApi(token=token)
    approved = release_approved_models(token)
    space_files = set(api.list_repo_files(repo_id=SPACE_ID, repo_type="space"))

    stale = []
    for path in space_files:
        if path == "models/training_artifact_sync.json":
            stale.append(path)
            continue
        parts = path.split("/")
        if len(parts) >= 4 and parts[0] == "artifacts" and parts[1] == "models":
            if parts[2] in MODELS and parts[2] not in approved:
                stale.append(path)

    if not stale:
        print("SPACE_ARTIFACT_CLEANUP=NOOP")
        print(f"RELEASE_APPROVED_MODELS={len(approved)}")
        print("UNAPPROVED_CHECKPOINTS_REMAINING=0")
        return

    operations = [CommitOperationDelete(path_in_repo=path) for path in sorted(stale)]
    commit = api.create_commit(
        repo_id=SPACE_ID, repo_type="space", operations=operations,
        commit_message=(
            "Remove unapproved AETHRION checkpoints from Space; "
            "retain canonical copies in private artifact repository"
        ),
    )
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError("Space cleanup returned no verifiable commit SHA.")
    current_files = set(api.list_repo_files(repo_id=SPACE_ID, repo_type="space"))
    remaining = [path for path in stale if path in current_files]
    if remaining:
        raise RuntimeError("Space cleanup verification failed: " + ", ".join(remaining[:10]))
    current = api.repo_info(repo_id=SPACE_ID, repo_type="space")
    if getattr(current, "sha", None) != revision:
        raise RuntimeError("Space HEAD does not match the cleanup commit.")
    print(f"SPACE_ARTIFACT_CLEANUP_COMMIT={revision}")
    print(f"SPACE_FILES_REMOVED={len(stale)}")
    print(f"RELEASE_APPROVED_MODELS_RETAINED={len(approved)}")
    print("PRIVATE_ARTIFACT_REPOSITORY=UNCHANGED")


if __name__ == "__main__":
    main()
