"""Owner-only experimental loader for the attested Chat candidate.

This path deliberately bypasses release registration only for a clearly labelled
private preview. It never changes models/manifest.json or marks a model released.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from aethrion.checkpoint import load_checkpoint
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer


class ExperimentalChatError(RuntimeError):
    """Candidate artifact failed preview-specific validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experimental_chat(root: str | Path, device: str | torch.device | None = None):
    """Load only the synced Chat candidate after validating its attested hashes."""
    root = Path(root).resolve()
    sync_path = root / "models" / "training_artifact_sync.json"
    manifest_path = root / "models" / "manifest.json"
    try:
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentalChatError("Chat preview manifests are missing or invalid.") from exc

    if sync.get("format") != "aethrion-training-artifact-sync-v1":
        raise ExperimentalChatError("Unsupported artifact sync manifest.")
    if sync.get("target_space") != "vizyon2027/AETHRION-LLM":
        raise ExperimentalChatError("Artifact sync manifest targets a different Space.")
    entry = sync.get("models", {}).get("chat")
    model_entry = manifest.get("models", {}).get("chat")
    if not isinstance(entry, dict) or not isinstance(model_entry, dict):
        raise ExperimentalChatError("Chat candidate is not present in both manifests.")
    if entry.get("status") != "checkpoint_available_not_released":
        raise ExperimentalChatError("Chat candidate is not marked as an unreleased synced checkpoint.")
    if entry.get("release_approval") is not False or sync.get("release_status_changed") is not False:
        raise ExperimentalChatError("Preview refuses any sync that changes release approval.")
    if model_entry.get("status") != "not_trained":
        raise ExperimentalChatError("Preview requires the production Chat status to remain not_trained.")

    checkpoint = root / "artifacts" / "models" / "chat" / "checkpoint_last.pt"
    tokenizer_path = root / "artifacts" / "models" / "chat" / "tokenizer.json"
    expected_checkpoint = entry.get("checkpoint_sha256")
    expected_tokenizer = entry.get("tokenizer_sha256")
    if not isinstance(expected_checkpoint, str) or len(expected_checkpoint) != 64:
        raise ExperimentalChatError("Chat checkpoint attestation is missing.")
    if not isinstance(expected_tokenizer, str) or len(expected_tokenizer) != 64:
        raise ExperimentalChatError("Chat tokenizer attestation is missing.")
    if not checkpoint.is_file() or _sha256(checkpoint) != expected_checkpoint:
        raise ExperimentalChatError("Chat checkpoint SHA-256 does not match the sync manifest.")
    if not tokenizer_path.is_file() or _sha256(tokenizer_path) != expected_tokenizer:
        raise ExperimentalChatError("Chat tokenizer SHA-256 does not match the sync manifest.")

    # load_checkpoint verifies the adjacent SHA sidecar and uses weights_only=True.
    state = load_checkpoint(checkpoint, map_location="cpu")
    config_data = state.get("model_config")
    if not isinstance(config_data, dict) or config_data != model_entry.get("model_config"):
        raise ExperimentalChatError("Checkpoint model configuration differs from the Space manifest.")
    if state.get("tokenizer_sha256") != expected_tokenizer:
        raise ExperimentalChatError("Checkpoint tokenizer hash differs from the synced tokenizer.")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    config = ModelConfig(**config_data)
    if tokenizer.vocab_size != config.vocab_size:
        raise ExperimentalChatError("Tokenizer vocabulary does not match the Chat model configuration.")
    weights = state.get("model")
    if not isinstance(weights, dict):
        raise ExperimentalChatError("Chat checkpoint does not contain a model state dictionary.")

    model = AETHRIONLM(config)
    try:
        model.load_state_dict(weights, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ExperimentalChatError("Chat checkpoint weights do not match AETHRIONLM.") from exc
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    return model, tokenizer, {
        "step": state.get("step"),
        "checkpoint_sha256": expected_checkpoint,
        "tokenizer_sha256": expected_tokenizer,
        "device": str(target_device),
        "status": "experimental_candidate_not_released",
    }


@torch.no_grad()
def generate_experimental_chat(
    model: AETHRIONLM,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty prompt.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer.")
    if not 1 <= max_new_tokens <= 64:
        raise ValueError("Experimental preview is limited to 1–64 new tokens.")
    ids = tokenizer.encode(prompt)[-model.config.block_size:]
    if not ids:
        raise ValueError("Prompt produced no tokenizer tokens.")
    context = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    output = model.generate(
        context, max_new_tokens=max_new_tokens, temperature=0.8,
        top_k=40, top_p=0.9, repetition_penalty=1.05,
    )
    return tokenizer.decode(output[0, context.size(1):].tolist())
