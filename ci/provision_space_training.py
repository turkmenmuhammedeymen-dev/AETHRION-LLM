"""Idempotently prepare the private artifact repo + Space settings for training sessions (run from CircleCI).
- creates <user>/aethrion-artifacts as a PRIVATE model repo if missing (never changes visibility of an existing one)
- sets the Space variable AETHRION_ARTIFACT_REPO
- never copies any token: the owner adds the Space secret HF_TOKEN (narrow, repo-limited token) in the Hugging Face UI
Prints booleans/ids only, never a token value."""
from __future__ import annotations
import os
import sys

SPACE_ID = "vizyon2027/AETHRION-LLM"
REPO_NAME = "aethrion-artifacts"


def provision(api, token: str, *, space_id: str = SPACE_ID, repo_name: str = REPO_NAME, share_token: bool = False) -> dict:
    user = api.whoami()
    owner = user.get("name") or ""
    if not owner:
        raise RuntimeError("cannot determine the Hugging Face account name from the token")
    repo_id = f"{owner}/{repo_name}"
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise RuntimeError(f"{repo_id} is not private; refusing to use it for checkpoints")
    current = api.get_space_variables(space_id) or {}
    existing = current.get("AETHRION_ARTIFACT_REPO")
    if (getattr(existing, "value", existing)) != repo_id:  # avoid needless Space restarts
        api.add_space_variable(space_id, "AETHRION_ARTIFACT_REPO", repo_id)
    runtime = api.get_space_runtime(space_id)
    existing_secrets = set(getattr(runtime, "secrets", None) or [])
    token_secret = "existing" if "HF_TOKEN" in existing_secrets else "missing"
    if token_secret == "missing" and share_token:
        api.add_space_secret(space_id, "HF_TOKEN", token, description="AETHRION artifact sync (provisioned by CI)")
        token_secret = "set-by-ci"
    return {"artifact_repo": repo_id, "private": True, "hf_token_secret": token_secret}


def main() -> int:
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in CircleCI context", file=sys.stderr)
        return 1
    result = provision(HfApi(token=token), token, share_token=False)
    print("PROVISIONED", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
