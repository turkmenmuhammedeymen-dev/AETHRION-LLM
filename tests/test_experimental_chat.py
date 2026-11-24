import hashlib
import json
from pathlib import Path

import pytest
import torch

from aethrion.checkpoint import save_checkpoint
from aethrion.experimental_chat import (
    ExperimentalChatError,
    generate_experimental_chat,
    load_experimental_chat,
)
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_root(root: Path) -> Path:
    model_dir = root / "artifacts" / "models" / "chat"
    model_dir.mkdir(parents=True)
    config = ModelConfig(
        vocab_size=256, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
    )
    model = AETHRIONLM(config)
    tokenizer_path = model_dir / "tokenizer.json"
    ByteBPETokenizer().save(tokenizer_path)
    checkpoint_path = model_dir / "checkpoint_last.pt"
    save_checkpoint({
        "format_version": 4,
        "step": 2999,
        "model": model.state_dict(),
        "optimizer": {},
        "scaler": {},
        "model_config": {
            "vocab_size": 256, "block_size": 16, "n_layer": 1,
            "n_head": 2, "n_embd": 16, "dropout": 0.0,
        },
        "data_sha256": "a" * 64,
        "tokenizer_sha256": _sha(tokenizer_path),
        "rng_state": {},
        "training_config": {"test_fixture": True},
    }, checkpoint_path)
    (root / "models").mkdir()
    (root / "models" / "manifest.json").write_text(json.dumps({
        "format": "aethrion-native-model-family-v1",
        "execution_policy": "local_only",
        "models": {"chat": {
            "architecture": "aethrion.model:AETHRIONLM",
            "model_config": {
                "vocab_size": 256, "block_size": 16, "n_layer": 1,
                "n_head": 2, "n_embd": 16, "dropout": 0.0,
            },
            "checkpoint": "artifacts/models/chat/checkpoint_last.pt",
            "tokenizer": "artifacts/models/chat/tokenizer.json",
            "status": "not_trained",
        }},
    }), encoding="utf-8")
    (root / "models" / "training_artifact_sync.json").write_text(json.dumps({
        "format": "aethrion-training-artifact-sync-v1",
        "target_space": "vizyon2027/AETHRION-LLM",
        "release_status_changed": False,
        "models": {"chat": {
            "status": "checkpoint_available_not_released",
            "checkpoint_sha256": _sha(checkpoint_path),
            "tokenizer_sha256": _sha(tokenizer_path),
            "release_approval": False,
        }},
    }), encoding="utf-8")
    return root


def test_load_and_generate_attested_chat_candidate(tmp_path):
    root = _candidate_root(tmp_path)
    model, tokenizer, info = load_experimental_chat(root, device="cpu")
    assert info["step"] == 2999
    assert info["status"] == "experimental_candidate_not_released"
    assert isinstance(generate_experimental_chat(model, tokenizer, "Hello", max_new_tokens=1), str)


def test_rejects_checkpoint_hash_mismatch(tmp_path):
    root = _candidate_root(tmp_path)
    checkpoint = root / "artifacts/models/chat/checkpoint_last.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ExperimentalChatError, match="checkpoint SHA-256"):
        load_experimental_chat(root, device="cpu")


def test_rejects_production_manifest_marked_trained(tmp_path):
    root = _candidate_root(tmp_path)
    path = root / "models/manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["chat"]["status"] = "trained"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentalChatError, match="remain not_trained"):
        load_experimental_chat(root, device="cpu")


def test_rejects_release_approval_in_sync_manifest(tmp_path):
    root = _candidate_root(tmp_path)
    path = root / "models/training_artifact_sync.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["chat"]["release_approval"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentalChatError, match="release approval"):
        load_experimental_chat(root, device="cpu")
