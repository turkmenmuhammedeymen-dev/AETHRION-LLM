"""Approval-gated response-only instruction tuning for AETHRION-BASE."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch

from aethrion.checkpoint import load_checkpoint, save_checkpoint, sha256_file, write_json_atomic
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.specialist_data import validate_specialist_dataset
from aethrion.training_config import require_finite_number, require_int, validate_numeric_tree


_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Stop requested (signal {signum}); instruction tuning will checkpoint after this step.", flush=True)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _batch(arrays, indices, device):
    return (
        torch.as_tensor(np.asarray(arrays["input_ids"][indices]), dtype=torch.long, device=device),
        torch.as_tensor(np.asarray(arrays["labels"][indices]), dtype=torch.long, device=device),
    )


@torch.no_grad()
def _evaluate(model, arrays, indices, device, batch_size):
    was_training = model.training
    model.eval()
    losses = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        inputs, labels = _batch(arrays, selected, device)
        _, loss = model(inputs, labels)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite instruction-tuning evaluation loss.")
        losses.append(float(loss.item()))
    model.train(was_training)
    if not losses:
        raise ValueError("Instruction-tuning evaluation split is empty.")
    return sum(losses) / len(losses)


def tune_instructions(
    config: dict[str, Any],
    manifest_path: str | Path,
    base_checkpoint: str | Path,
    *,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    required = {
        "model_config", "tokenizer_path", "output_dir", "seed", "max_steps",
        "batch_size", "learning_rate", "weight_decay", "grad_clip",
        "eval_interval", "save_interval",
    }
    missing = required - set(config)
    if missing:
        raise ValueError("Missing instruction-tuning config keys: " + ", ".join(sorted(missing)))
    for key in ("max_steps", "batch_size", "eval_interval", "save_interval"):
        require_int(config[key], key, minimum=1)
    require_int(config["seed"], "seed", minimum=0)
    require_finite_number(config["learning_rate"], "learning_rate", minimum=0, minimum_inclusive=False)
    require_finite_number(config["weight_decay"], "weight_decay", minimum=0)
    require_finite_number(config["grad_clip"], "grad_clip", minimum=0, minimum_inclusive=False)
    validate_numeric_tree(config["model_config"])
    if not isinstance(config["tokenizer_path"], str) or not config["tokenizer_path"].strip():
        raise ValueError("tokenizer_path must be a non-empty path string.")
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a non-empty path string.")

    approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
    if approval["model_id"] != "chat":
        raise ValueError("AETHRION-BASE instruction tuning requires model_id='chat'.")
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_file)
    tokenizer_path = Path(approval["tokenizer_path"]).resolve()
    configured_tokenizer_path = Path(config["tokenizer_path"]).resolve()
    if not tokenizer_path.is_file() or not configured_tokenizer_path.is_file():
        raise ValueError("Instruction-tuning tokenizer is missing.")
    tokenizer_hash = sha256_file(tokenizer_path)
    if (
        manifest.get("tokenizer_sha256") != tokenizer_hash
        or sha256_file(configured_tokenizer_path) != tokenizer_hash
    ):
        raise ValueError("Instruction tokenizer must byte-match both the reviewed manifest copy and the BASE tokenizer configured for this run.")

    base = load_checkpoint(base_checkpoint, map_location="cpu")
    model_config = ModelConfig(**config["model_config"])
    model_config.validate()
    if base.get("model_config") != model_config.__dict__:
        raise ValueError("Base checkpoint model configuration differs from instruction-tuning config.")
    if base.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Base checkpoint tokenizer does not match instruction-tuning tokenizer.")

    with np.load(approval["dataset_path"], allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    ids, labels, split = arrays["input_ids"], arrays["labels"], arrays["split"].astype(str)
    if ids.ndim != 2 or labels.shape != ids.shape or ids.shape[0] != len(split):
        raise ValueError("Instruction dataset input_ids and labels must have matching [samples, sequence] shapes.")
    if ids.shape[1] > model_config.block_size:
        raise ValueError("Instruction sequence length exceeds base model block_size.")
    if ids.min() < 0 or ids.max() >= model_config.vocab_size:
        raise ValueError("Instruction input token ID is outside the base model vocabulary.")
    supervised = labels != -100
    if np.any(~supervised.any(axis=1)):
        raise ValueError("Every instruction example must contain at least one supervised response token.")
    valid_labels = labels[supervised]
    if valid_labels.min() < 0 or valid_labels.max() >= model_config.vocab_size:
        raise ValueError("Instruction labels must be valid token IDs or the -100 ignore index.")

    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
        raise ValueError("Instruction dataset requires train, validation and test examples.")

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AETHRIONLM(model_config).to(device)
    model.load_state_dict(base["model"])
    base_checkpoint_hash = sha256_file(base_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )
    start_step = 0
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_last.pt"
    if resume is not None:
        state = load_checkpoint(resume, map_location=device)
        if state.get("model_config") != model_config.__dict__:
            raise ValueError("Resume checkpoint model config differs.")
        if state.get("data_sha256") != approval["dataset_sha256"]:
            raise ValueError("Instruction dataset changed; refusing unsafe resume.")
        if state.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("Instruction tokenizer changed; refusing unsafe resume.")
        if state.get("manifest_sha256") != manifest_hash:
            raise ValueError("Instruction source/approval manifest changed; refusing unsafe resume.")
        if state.get("base_checkpoint_sha256") != base_checkpoint_hash:
            raise ValueError("Base checkpoint changed; refusing unsafe instruction-tuning resume.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("rng_state"):
            random.setstate(state["rng_state"]["python"])
            np.random.set_state(state["rng_state"]["numpy"])
            torch.set_rng_state(state["rng_state"]["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in state["rng_state"]:
                torch.cuda.set_rng_state_all([item.cpu() for item in state["rng_state"]["cuda"]])
        checkpoint_step = state.get("step")
        if type(checkpoint_step) is not int or checkpoint_step < 0:
            raise ValueError("Instruction-tuning resume checkpoint step is invalid.")
        start_step = checkpoint_step + 1
        if start_step >= config["max_steps"]:
            raise ValueError("max_steps must exceed the instruction-tuning resume checkpoint step.")

    started = time.perf_counter()
    metrics_path = output / "metrics.jsonl"
    validation_loss = None
    model.train()
    for step in range(start_step, config["max_steps"]):
        selected = np.random.choice(train_indices, size=config["batch_size"], replace=True)
        inputs, targets = _batch(arrays, selected, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite instruction-tuning loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()

        if step % config["eval_interval"] == 0 or step == config["max_steps"] - 1 or _STOP_REQUESTED:
            validation_loss = _evaluate(
                model, arrays, val_indices, device, config["batch_size"]
            )
            record = {
                "stage": "instruction_tuning", "step": step,
                "train_loss": float(loss.item()), "validation_loss": validation_loss,
                "dataset_sha256": approval["dataset_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(record, ensure_ascii=False), flush=True)

        if step % config["save_interval"] == 0 or step == config["max_steps"] - 1:
            payload = {
                "format_version": 3,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": {},
                "model_config": model_config.__dict__,
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_hash,
                "data_path": approval["dataset_path"],
                "data_sha256": approval["dataset_sha256"],
                "manifest_sha256": manifest_hash,
                "rng_state": _rng_state(),
                "training_stage": "instruction_tuning",
                "base_checkpoint_sha256": base_checkpoint_hash,
            }
            checkpoint_hash = save_checkpoint(payload, checkpoint_path)
            write_json_atomic({
                "status": "instruction_tuning_in_progress_not_released",
                "model_id": "chat",
                "release_status": "NOT_RELEASED",
                "checkpoint_sha256": checkpoint_hash,
                "dataset_sha256": approval["dataset_sha256"],
                "tokenizer_sha256": tokenizer_hash,
                "manifest_sha256": manifest_hash,
                "validation_loss": validation_loss,
                "note": "Independent benchmark, safety review and explicit release approval remain required."
            }, output / "instruction_tuning_report.json")

        if _STOP_REQUESTED and step < config["max_steps"] - 1:
            interrupted_report = {
                "status": "instruction_tuning_interrupted_not_released",
                "model_id": "chat",
                "release_status": "NOT_RELEASED",
                "steps": step + 1,
                "dataset_sha256": approval["dataset_sha256"],
                "tokenizer_sha256": tokenizer_hash,
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
                "validation_loss": validation_loss,
                "device": str(device),
                "note": "Instruction tuning was safely paused after a verified checkpoint; resume with the same base checkpoint, dataset and tokenizer."
            }
            write_json_atomic(interrupted_report, output / "instruction_tuning_report.json")
            return interrupted_report

    test_loss = _evaluate(model, arrays, test_indices, device, config["batch_size"])
    report = {
        "status": "instruction_tuning_complete_not_released",
        "model_id": "chat",
        "release_status": "NOT_RELEASED",
        "steps": config["max_steps"],
        "dataset_sha256": approval["dataset_sha256"],
        "tokenizer_sha256": tokenizer_hash,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_loss": validation_loss,
        "held_out_test_loss": test_loss,
        "sample_counts": approval["split_counts"],
        "device": str(device),
        "note": "A passing loss value is not proof of instruction-following quality or release readiness."
    }
    write_json_atomic(report, output / "instruction_tuning_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Instruction-tune AETHRION-BASE from a verified base checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous_handlers = {}
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[stop_signal] = signal.signal(stop_signal, _request_stop)
    try:
        report = tune_instructions(config, args.manifest, args.base_checkpoint, resume=args.resume)
    finally:
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
