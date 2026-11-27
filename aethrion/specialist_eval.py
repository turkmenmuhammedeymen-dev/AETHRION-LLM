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


def _research_metrics(model, arrays, idx, device) -> dict[str, Any]:
    """Retrieval over the held-out test passages: top-1 / MRR versus random ranking."""
    idx = np.asarray(idx)
    if len(idx) < 2:
        return {}
    model.eval()
    def _ids(key):
        ids = torch.from_numpy(np.asarray(arrays[key][idx]).copy()).long().to(device)
        mask = ids != 0
        mask[:, 0] = True
        length = max(1, int(mask.sum(1).max()))
        return ids[:, :length], mask[:, :length]
    (qi, qm), (pi, pm) = _ids("query_ids"), _ids("passage_ids")
    with torch.no_grad():
        scores = model.score_matrix(qi, pi, qm, pm)[2].cpu().numpy()
    groups = np.asarray(arrays["group_id"][idx]).astype(str)
    ranks = []
    for i in range(len(idx)):
        order = np.argsort(-scores[i])
        ranks.append(next(r for r, j in enumerate(order, 1) if i == j or groups[j] == groups[i]))
    ranks = np.asarray(ranks, dtype=np.float64)
    n = len(idx)
    return {"task_metrics": {"accuracy": float((ranks == 1).mean()), "majority_baseline_accuracy": float(1.0 / n),
                             "mrr": float((1.0 / ranks).mean()), "support": int(n)}}


def _vision_metrics(model, arrays, idx, device) -> dict[str, Any]:
    """Image->caption retrieval on held-out items; same-group captions count as correct."""
    idx = np.asarray(idx)
    if len(idx) < 2:
        return {}
    model.eval()
    ids = torch.from_numpy(np.asarray(arrays["token_ids"][idx]).copy()).long().to(device)
    mask = ids != 0
    mask[:, 0] = True
    keep = max(1, int(mask.sum(1).max()))
    imgs = torch.from_numpy(np.asarray(arrays["images"][idx]).copy()).to(device).float().div(127.5).sub(1.0)
    with torch.no_grad():
        scores = (model.encode_image(imgs) @ model.encode_text(ids[:, :keep], mask[:, :keep]).T).cpu().numpy()
    groups = np.asarray(arrays["group_id"][idx]).astype(str)
    ranks = []
    for i in range(len(idx)):
        order = np.argsort(-scores[i])
        ranks.append(next(r for r, j in enumerate(order, 1) if i == j or groups[j] == groups[i]))
    ranks = np.asarray(ranks, dtype=np.float64)
    n = len(idx)
    return {"task_metrics": {"accuracy": float((ranks == 1).mean()), "majority_baseline_accuracy": float(1.0 / n),
                             "mrr": float((1.0 / ranks).mean()), "support": int(n)}}


def _split_prompt(text: str) -> str | None:
    """Text up to the end of the first function's signature/docstring; None if there is no function."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = node.body
            cut = body[0].end_lineno if (isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                                         and isinstance(body[0].value.value, str)) else node.lineno
            lines = text.splitlines()[:cut]
            return "\n".join(lines) + "\n"
    return None


def _code_generation_metrics(model, tokenizer_path, arrays, idx, device, max_new: int = 160) -> dict[str, Any]:
    """Greedy-complete each held-out function from its signature+docstring and check that
    prompt+completion is valid Python that defines the function (compile + exec of the definition)."""
    import ast
    from aethrion.tokenizer import ByteBPETokenizer
    tok = ByteBPETokenizer.load(tokenizer_path)
    model.eval()
    total = parsed = defined = 0
    for i in np.asarray(idx):
        ids = [int(t) for t in np.asarray(arrays["token_ids"][i]) if int(t) != 0]
        prompt = _split_prompt(tok.decode(ids))
        if not prompt:
            continue
        pids = tok.encode(prompt)[: model.config.block_size - max_new]
        out = model.generate(torch.tensor([pids], dtype=torch.long, device=device), max_new, temperature=1e-4, top_k=1,
                             top_p=1.0, repetition_penalty=1.0)
        new = tok.decode([int(t) for t in out[0].tolist()[len(pids):]])
        total += 1
        code = prompt + new
        tree = None
        # trim to the longest valid prefix at a line boundary (the model may run on into unrelated text)
        lines = code.splitlines()
        for end in range(len(lines), 0, -1):
            try:
                tree = ast.parse("\n".join(lines[:end]) + "\n")
                break
            except SyntaxError:
                continue
        if tree is None or not any(isinstance(n, ast.FunctionDef) for n in tree.body):
            continue
        parsed += 1
        try:
            ns: dict[str, Any] = {}
            exec(compile(ast.Module([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], []), "<gen>", "exec"), {"__builtins__": __builtins__}, ns)
            defined += 1
        except Exception:
            pass
    if not total:
        return {}
    return {"code_generation": {"functions": total, "valid_python_rate": parsed / total, "defines_function_rate": defined / total,
                                "note": "Syntax/definition validity only; behavioural pass@k needs per-task unit tests."}}


def _task_metrics(model, model_id, arrays, keys, idx, device, batch_size) -> dict[str, Any]:
    if model_id == "research":
        return _research_metrics(model, arrays, idx, device)
    if model_id == "vision":
        return _vision_metrics(model, arrays, idx, device)
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
    if model_id == "code" and approval.get("tokenizer_path"):
        report.update(_code_generation_metrics(trained, approval["tokenizer_path"], arrays, test_idx, dev))
    # Untrained model on the same metric: shows whether training adds anything beyond the architecture itself.
    base_task = _task_metrics(baseline, model_id, arrays, keys, test_idx, dev, batch_size)
    if base_task:
        report["untrained_baseline_task_metrics"] = base_task["task_metrics"]
    return report
