"""Run a disposable forward/backward smoke check for all 11 native architectures.

This is an architecture diagnostic only: it uses in-memory synthetic tensors,
never reads a dataset, never calls optimizer.step(), and never writes checkpoints.
It does not train, approve, or release any AETHRION model.
"""
from __future__ import annotations

import argparse
import gc
import json
from typing import Any

import torch

from aethrion.specialist_train import _loss
from aethrion.zerogpu_preflight import (
    MODEL_IDS,
    _build_preflight_model,
    _synthetic_batch,
)


def run_preflight(device_name: str = "auto") -> dict[str, Any]:
    """Exercise every model's scalar loss and gradients on CPU or CUDA."""
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda:0" if device_name == "cuda" or
        (device_name == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results = []
    failures = []
    for model_id in MODEL_IDS:
        model = None
        try:
            model = _build_preflight_model(model_id, device)
            if model_id == "chat":
                token_ids = torch.randint(0, 2048, (1, 16), device=device)
                _, loss = model(token_ids, token_ids.clone())
            else:
                loss = _loss(model, model_id, _synthetic_batch(model_id, device))
            if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise RuntimeError("Model returned a missing, non-scalar, or non-finite loss.")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Model produced missing or non-finite gradients.")
            results.append({
                "model": model_id,
                "status": "PASS",
                "parameters": sum(p.numel() for p in model.parameters()),
                "synthetic_loss": round(float(loss.detach().item()), 6),
            })
        except Exception as exc:
            failure = f"{type(exc).__name__}: {str(exc)[:240]}"
            failures.append({"model": model_id, "error": failure})
            results.append({"model": model_id, "status": "FAIL", "error": failure})
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        total_memory_gib = round(properties.total_memory / (1024 ** 3), 2)
        peak_allocated_gib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2)
        device_label = torch.cuda.get_device_name(device)
    else:
        total_memory_gib = None
        peak_allocated_gib = None
        device_label = "CPU"

    return {
        "status": "PASS" if not failures else "PARTIAL_FAILURE",
        "device": device_label,
        "total_memory_gib": total_memory_gib,
        "peak_allocated_gib": peak_allocated_gib,
        "models_tested": len(results),
        "models_passed": sum(row["status"] == "PASS" for row in results),
        "models_failed": len(failures),
        "results": results,
        "failures": failures,
        "synthetic_tensors_only": True,
        "dataset_used": False,
        "optimizer_steps": 0,
        "checkpoints_written": False,
        "training_started": False,
        "release_status": "NOT_RELEASED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run_preflight(args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
