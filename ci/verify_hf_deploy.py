import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing from the aethrion-deploy CircleCI context"
        )

    repo_id = "vizyon2027/AETHRION-LLM"
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path="hf_space",
        repo_id=repo_id,
        repo_type="space",
        commit_message=(
            "Deploy complete AETHRION-LLM source, configs, docs, "
            "data governance and tests"
        ),
        ignore_patterns=[
            "README.md",
            "**/__pycache__/**",
            "**/*.pyc",
            ".DS_Store",
        ],
    )
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError(
            "Hugging Face upload did not return a verifiable commit SHA"
        )

    # Verify the complete runtime and every model's training configuration,
    # not merely the UI and top-level trainer entrypoints.
    required_files = [
        "app.py",
        "pyproject.toml",
        "aethrion/space_training.py",
        "aethrion/checkpoint.py",
        "aethrion/model.py",
        "aethrion/tokenizer.py",
        "aethrion/training_config.py",
        "aethrion/family_audit.py",
        "aethrion/artifact_semantics.py",
        "aethrion/zerogpu_preflight.py",
        "aethrion/space_auth.py",
        "aethrion/train.py",
        "aethrion/train_family.py",
        "aethrion/specialist_train.py",
        "aethrion/specialist_data.py",
        "aethrion/prepare_data.py",
        "configs/small.json",
        "configs/specialists/code.json",
        "configs/specialists/vision.json",
        "configs/specialists/image.json",
        "configs/specialists/asr.json",
        "configs/specialists/tts.json",
        "configs/specialists/vocoder.json",
        "configs/specialists/3d.json",
        "configs/specialists/research.json",
        "configs/specialists/document.json",
        "configs/specialists/safety.json",
        "data/base_sources.template.json",
        "data/processed/corpus_manifest.json",
        "models/manifest.json",
    ]
    for relative_path in required_files:
        local_path = Path("hf_space") / relative_path
        remote_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=relative_path,
                repo_type="space",
                revision=revision,
                token=token,
            )
        )
        local_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
        remote_hash = hashlib.sha256(remote_path.read_bytes()).hexdigest()
        if local_hash != remote_hash:
            raise RuntimeError(
                f"Hugging Face read-back SHA-256 mismatch: {relative_path}"
            )
        print(f"Verified deployed file: {relative_path} ({remote_hash})")

    print(
        "AETHRION-LLM deployment verified at Hugging Face commit "
        f"{revision}."
    )


if __name__ == "__main__":
    main()
