"""Read-only held-out evaluation of a specialist checkpoint (never trains, never releases).

Compares the trained checkpoint with a freshly initialised model built from the same
seeded config on the held-out *test* split, and adds task metrics where the output
format allows (safety decision accuracy / macro-F1 vs majority baseline, document tag
accuracy, code next-token accuracy). Result status is always ``evaluated_not_released``;
``evidence_of_learning`` is a measurement, not a pass/release decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aethrion.specialist_data import validate_specialist_dataset
from aethrion.specialist_train import (_build_model, _evaluate, _load_checkpoint, _to_batch,
                                       _validate_model_data_compatibility)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = int(((y_pred == c) & (y_true == c)).sum()); fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _task_metrics(model, model_id, arrays, keys, idx, device, batch_size) -> dict[str, Any]:
    if model_id not in {"safety", "document", "code"}:
        return {}
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = _to_batch(arrays, keys, idx[start:start + batch_size], device)
            if model_id == "safety":
                out = model(batch["token_ids"], batch.get("attention_mask"))
                ys.append(batch["decision_labels"].cpu().numpy()); ps.append(out["decision_logits"].argmax(-1).cpu().numpy())
            elif model_id == "document":
                out = model(batch["token_ids"], batch["boxes"], batch["layout_ids"], batch.get("attention_mask"))
                labels = batch["tag_labels"]; mask = labels != -100
                ys.append(labels[mask].cpu().numpy()); ps.append(out["tag_logits"].argmax(-1)[mask].cpu().numpy())
            else:
                ids = batch["token_ids"]; logits = model(ids[:, :-1], ids[:, 1:])[0]
                ys.append(ids[:, 1:].reshape(-1).cpu().numpy()); ps.append(logits.argmax(-1).reshape(-1).cpu().numpy())
    if not ys:
        return {}
    y = np.concatenate(ys); p = np.concatenate(ps)
    if not len(y):
        return {}
    majority = int(np.bincount(y.astype(np.int64)).argmax())
    metrics = {"accuracy": float((y == p).mean()), "majority_baseline_accuracy": float((y == majority).mean()),
               "support": int(len(y))}
    if model_id in {"safety", "document"}:
        metrics["macro_f1"] = _macro_f1(y, p)
    return {"task_metrics": metrics}


def evaluate_specialist(config: dict[str, Any], manifest_path: str | Path, checkpoint_path: str | Path,
                        *, device: str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    model_id = config["model_id"]
    approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
    if approval["model_id"] != model_id:
        raise ValueError("Config model_id differs from approved dataset manifest.")
    state = _load_checkpoint(Path(checkpoint_path), dev)  # sha256 sidecar + weights_only
    if state["model_id"] != model_id or state["dataset_sha256"] != approval["dataset_sha256"]:
        raise ValueError("Checkpoint does not match the approved dataset for this model.")
    if state["model_config"] != config["model_config"]:
        raise ValueError("Checkpoint model configuration differs from config.")
    with np.load(approval["dataset_path"], allow_pickle=False) as archive:
        arrays = {k: np.array(archive[k], copy=True) for k in archive.files}
    _validate_model_data_compatibility(model_id, arrays, config["model_config"])
    split = arrays["split"].astype(str)
    test_idx = np.flatnonzero(split == "test")
    if not len(test_idx):
        raise ValueError("Test partition is empty.")
    keys = sorted(set(arrays) - {"sample_id", "group_id", "split"})
    batch_size = int(config["batch_size"])

    torch.manual_seed(int(config["seed"]))
    baseline = _build_model(model_id, config["model_config"]).to(dev)
    trained = _build_model(model_id, config["model_config"]).to(dev)
    trained.load_state_dict(state["model_state"])
    n_params = int(sum(p.numel() for p in trained.parameters()))
    finite = all(bool(torch.isfinite(p).all()) for p in trained.parameters())
    base_loss = _evaluate(baseline, model_id, arrays, keys, test_idx, dev, batch_size)
    test_loss = _evaluate(trained, model_id, arrays, keys, test_idx, dev, batch_size)
    rel = (base_loss - test_loss) / base_loss if base_loss else 0.0
    report = {
        "status": "evaluated_not_released", "release_status": "NOT_RELEASED", "model_id": model_id,
        "checkpoint_step": int(state["step"]), "max_steps": int(config["max_steps"]),
        "training_complete": int(state["step"]) + 1 >= int(config["max_steps"]),
        "parameters": n_params, "weights_finite": finite, "test_samples": int(len(test_idx)),
        "test_loss": test_loss, "untrained_baseline_test_loss": base_loss,
        "relative_loss_improvement": rel, "evidence_of_learning": bool(finite and rel > 0.05),
        "dataset_sha256": approval["dataset_sha256"], "checkpoint_dataset_sha256": state["dataset_sha256"],
        "note": ("Held-out loss on synthetic first-party data only. Not a release decision; "
                 "independent quality, safety review and human approval remain required."),
    }
    report.update(_task_metrics(trained, model_id, arrays, keys, test_idx, dev, batch_size))
    return report
