"""Short synthetic GPU architecture preflight; never trains on or saves user data."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.specialist_train import _build_model, _loss

# Keep in sync with configs/specialists/*.json. These are architecture settings,
# not data, checkpoints, pretrained weights, or production training parameters.
SPECIALIST_CONFIGS: dict[str, dict[str, Any]] = {
    "code": {"vocab_size": 2048, "block_size": 512, "n_layer": 4, "n_head": 4, "n_embd": 128, "dropout": 0.1},
    "vision": {"image_size": 224, "patch_size": 16, "vocab_size": 2048, "max_text_length": 128, "width": 128, "layers": 4, "heads": 4, "projection_dim": 256, "dropout": 0.1},
    "image": {"vocab_size": 2048, "image_size": 64, "text_length": 128, "width": 64, "time_dim": 128, "diffusion_steps": 1000},
    "asr": {"vocab_size": 2048, "n_mels": 80, "width": 128, "layers": 4, "heads": 4, "dropout": 0.1, "blank_id": 0},
    "tts": {"vocab_size": 2048, "mel_bins": 80, "width": 128, "layers": 4, "heads": 4, "dropout": 0.1},
    "vocoder": {"mel_bins": 80, "channels": 128, "hop_length": 256},
    "3d": {"vocab_size": 2048, "width": 128, "points": 512, "layers": 4, "heads": 4, "diffusion_steps": 1000},
    "research": {"vocab_size": 2048, "max_length": 512, "width": 128, "layers": 4, "heads": 4, "embedding_dim": 256},
    "document": {"vocab_size": 4096, "layout_vocab_size": 32, "max_length": 512, "width": 128, "layers": 4, "heads": 4, "tag_classes": 32},
    "safety": {"vocab_size": 4096, "max_length": 512, "width": 128, "layers": 4, "heads": 4, "reason_classes": 16, "decisions": 3},
}
MODEL_IDS = ("chat", *SPECIALIST_CONFIGS.keys())


def _synthetic_batch(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Create tiny, disposable tensors solely to exercise one model loss."""
    def ids(shape, vocab=2048):
        return torch.randint(0, vocab, shape, device=device, dtype=torch.long)

    if model_id == "code":
        return {"token_ids": ids((1, 16))}
    if model_id == "vision":
        return {"images": torch.randn(1, 3, 224, 224, device=device), "token_ids": ids((1, 8))}
    if model_id == "image":
        return {"images": torch.randn(1, 3, 64, 64, device=device), "token_ids": ids((1, 8))}
    if model_id == "asr":
        return {"features": torch.randn(1, 16, 80, device=device), "input_lengths": torch.tensor([16], device=device),
                "targets": torch.randint(1, 2048, (2,), device=device, dtype=torch.long), "target_lengths": torch.tensor([2], device=device)}
    if model_id == "tts":
        return {"token_ids": ids((1, 4)), "frame_to_token": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], device=device),
                "target_mels": torch.randn(1, 8, 80, device=device), "durations": torch.tensor([[2, 2, 2, 2]], device=device),
                "frame_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
                "token_mask": torch.ones(1, 4, dtype=torch.bool, device=device)}
    if model_id == "vocoder":
        return {"target_mels": torch.randn(1, 8, 80, device=device),
                "target_waveforms": torch.randn(1, 8 * 256, device=device),
                "waveform_mask": torch.ones(1, 8 * 256, dtype=torch.bool, device=device)}
    if model_id == "3d":
        return {"points": torch.randn(1, 512, 3, device=device), "token_ids": ids((1, 8))}
    if model_id == "research":
        return {"query_ids": ids((1, 8)), "passage_ids": ids((1, 8))}
    if model_id == "document":
        return {"token_ids": ids((1, 8), 4096), "boxes": torch.rand(1, 8, 4, device=device),
                "layout_ids": ids((1, 8), 32), "tag_labels": ids((1, 8), 32)}
    if model_id == "safety":
        return {"token_ids": ids((1, 8), 4096), "decision_labels": torch.tensor([1], device=device),
                "reason_labels": torch.zeros(1, 16, device=device)}
    raise ValueError(f"Unknown model: {model_id}")


def _build_preflight_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """Construct each native architecture directly on the allocated test device."""
    if model_id == "chat":
        model = AETHRIONLM(ModelConfig(vocab_size=2048, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.0))
    else:
        model = _build_model(model_id, SPECIALIST_CONFIGS[model_id])
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
                token_ids = torch.randint(0, 2048, (1, 16), device=device)
                targets = token_ids.clone()
                _, loss = model(token_ids, targets)
            else:
                batch = _synthetic_batch(model_id, device)
                loss = _loss(model, model_id, batch)
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
