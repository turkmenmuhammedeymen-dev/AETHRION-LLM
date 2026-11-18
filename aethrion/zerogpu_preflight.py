"""Short synthetic GPU architecture preflight; never trains on or saves user data."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.specialist_train import _build_model, _loss


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPOSITORY_ROOT / "models" / "manifest.json"


def _load_model_manifest() -> dict[str, Any]:
    """Read the canonical model configs; never maintain a second config copy."""
    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("models/manifest.json must contain a models object.")
    expected = {"chat", "code", "vision", "image", "asr", "tts", "vocoder",
                "3d", "research", "document", "safety"}
    if set(models) != expected:
        raise ValueError(
            "Native model manifest IDs differ from the required 11-model set: "
            + ", ".join(sorted(set(models) ^ expected))
        )
    for model_id, entry in models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("model_config"), dict):
            raise ValueError(f"Missing model_config for native model {model_id}.")
    return models


# These values are loaded from the same manifest used by model release routing.
MODEL_MANIFEST = _load_model_manifest()
MODEL_IDS = tuple(MODEL_MANIFEST.keys())
SPECIALIST_CONFIGS: dict[str, dict[str, Any]] = {
    model_id: entry["model_config"]
    for model_id, entry in MODEL_MANIFEST.items()
    if model_id != "chat"
}


def _synthetic_batch(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Create tiny, disposable tensors solely to exercise one model loss."""
    config = MODEL_MANIFEST[model_id]["model_config"]

    def ids(shape, vocab=None):
        size = int(vocab or config.get("vocab_size", 2048))
        return torch.randint(0, size, shape, device=device, dtype=torch.long)

    if model_id == "code":
        return {"token_ids": ids((1, 16))}
    if model_id == "vision":
        size = config["image_size"]
        return {"images": torch.randn(1, 3, size, size, device=device), "token_ids": ids((1, 8))}
    if model_id == "image":
        size = config["image_size"]
        return {"images": torch.randn(1, 3, size, size, device=device), "token_ids": ids((1, 8))}
    if model_id == "asr":
        return {
            "features": torch.randn(1, 16, config["n_mels"], device=device),
            "input_lengths": torch.tensor([16], device=device),
            "targets": torch.randint(1, config["vocab_size"], (1, 2), device=device, dtype=torch.long),
            "target_lengths": torch.tensor([2], device=device),
        }
    if model_id == "tts":
        return {
            "token_ids": ids((1, 4)),
            "frame_to_token": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], device=device),
            "target_mels": torch.randn(1, 8, config["mel_bins"], device=device),
            "durations": torch.tensor([[2, 2, 2, 2]], device=device),
            "frame_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
            "token_mask": torch.ones(1, 4, dtype=torch.bool, device=device),
        }
    if model_id == "vocoder":
        frames = 8
        samples = frames * config["hop_length"]
        return {
            "target_mels": torch.randn(1, frames, config["mel_bins"], device=device),
            "target_waveforms": torch.randn(1, samples, device=device),
            "waveform_mask": torch.ones(1, samples, dtype=torch.bool, device=device),
        }
    if model_id == "3d":
        return {"points": torch.randn(1, config["points"], 3, device=device), "token_ids": ids((1, 8))}
    if model_id == "research":
        return {"query_ids": ids((1, 8)), "passage_ids": ids((1, 8))}
    if model_id == "document":
        return {
            "token_ids": ids((1, 8)),
            "boxes": torch.rand(1, 8, 4, device=device),
            "layout_ids": ids((1, 8), config["layout_vocab_size"]),
            "tag_labels": ids((1, 8), config["tag_classes"]),
        }
    if model_id == "safety":
        return {
            "token_ids": ids((1, 8)),
            "decision_labels": torch.tensor([1], device=device),
            "reason_labels": torch.zeros(1, config["reason_classes"], device=device),
        }
    raise ValueError(f"Unknown model: {model_id}")


def _build_preflight_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """Construct each native architecture directly on the allocated test device."""
    config = dict(MODEL_MANIFEST[model_id]["model_config"])
    if model_id == "chat":
        model = AETHRIONLM(ModelConfig(**config))
    else:
        model = _build_model(model_id, config)
    return model.to(device)


def run_zero_gpu_preflight() -> dict[str, Any]:
    """Run one synthetic forward/backward per native model; never optimizer.step()."""
    if not torch.cuda.is_available():
        raise RuntimeError("ZeroGPU preflight requires an allocated CUDA device.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    results = []
    failures = []
    for model_id in MODEL_IDS:
        model = None
        try:
            model = _build_preflight_model(model_id, device)
            if model_id == "chat":
                vocab_size = int(MODEL_MANIFEST[model_id]["model_config"]["vocab_size"])
                token_ids = torch.randint(0, vocab_size, (1, 16), device=device)
                _, loss = model(token_ids, token_ids.clone())
            else:
                loss = _loss(model, model_id, _synthetic_batch(model_id, device))
            if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise RuntimeError("Model returned a missing, non-scalar, or non-finite loss.")
            loss.backward()
            grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
            if not grad_tensors or not all(torch.isfinite(g).all().item() for g in grad_tensors):
                raise RuntimeError("Model produced missing or non-finite gradients.")
            params = sum(p.numel() for p in model.parameters())
            results.append({"model": model_id, "status": "PASS", "parameters": params,
                            "synthetic_loss": round(float(loss.detach().item()), 6)})
        except Exception as exc:
            failures.append({"model": model_id, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
            results.append({"model": model_id, "status": "FAIL", "error": failures[-1]["error"]})
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "status": "PASS" if not failures else "PARTIAL_FAILURE",
        "device": torch.cuda.get_device_name(device),
        "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
        "models_tested": len(results),
        "models_passed": sum(row["status"] == "PASS" for row in results),
        "models_failed": len(failures),
        "results": results,
        "failures": failures,
        "training_started": False,
        "optimizer_steps": 0,
        "checkpoints_written": False,
        "dataset_used": False,
        "release_status": "NOT_RELEASED",
    }


if __name__ == "__main__":
    print(json.dumps(run_zero_gpu_preflight(), ensure_ascii=False, indent=2))
