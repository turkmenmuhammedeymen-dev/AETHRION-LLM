"""Training loop with strict data checks and reproducible resumable checkpoints."""
from __future__ import annotations
import argparse, json, math, random, signal, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer, validate_tokenizer_provenance
from aethrion.checkpoint import load_checkpoint, save_checkpoint, sha256_file, write_json_atomic
from aethrion.training_config import require_finite_number, require_int

_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Stop requested (signal {signum}); will save after the current training step.", flush=True)


def _base_resume_training_config(cfg: dict, schedule_max_steps: int) -> dict:
    """Persist the exact BASE training settings and original LR schedule horizon."""
    return {
        "stage": "base_pretraining",
        "seed": int(cfg["seed"]),
        "batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "min_learning_rate": float(cfg["min_learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "eval_interval": int(cfg["eval_interval"]),
        "eval_batches": int(cfg["eval_batches"]),
        "save_interval": int(cfg["save_interval"]),
        "grad_clip": float(cfg["grad_clip"]),
        "amp": bool(cfg["amp"]),
        "max_steps": int(cfg["max_steps"]),
        "schedule_max_steps": int(schedule_max_steps),
    }


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def capture_rng_state() -> dict:
    """Capture RNG state using weights-only-safe primitive containers and tensors."""
    py = random.getstate()
    np_state = np.random.get_state()
    state = {
        "python": {
            "version": int(py[0]),
            "state": [int(value) for value in py[1]],
            "gauss": py[2],
        },
        "numpy": {
            "algorithm": str(np_state[0]),
            "keys": np_state[1].tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore the primitive RNG representation saved in native checkpoints."""
    py = state["python"]
    random.setstate((int(py["version"]), tuple(int(value) for value in py["state"]), py["gauss"]))
    ns = state["numpy"]
    np.random.set_state((
        str(ns["algorithm"]), np.asarray(ns["keys"], dtype=np.uint32),
        int(ns["position"]), int(ns["has_gauss"]), float(ns["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def load_tokens(path: str, tokenizer: ByteBPETokenizer, *, min_tokens: int = 2) -> torch.Tensor:
    ids = tokenizer.encode(Path(path).read_text(encoding="utf-8"))
    if len(ids) < min_tokens:
        raise ValueError(f"Approved data split {path} is too small; requires at least {min_tokens} tokens.")
    return torch.tensor(ids, dtype=torch.long)

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if batch_size < 1 or block_size < 1: raise ValueError("batch_size and block_size must be positive")
    if len(data) <= block_size + 1: raise ValueError("Tokenized corpus must contain more tokens than block_size + 1")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    was_training = model.training
    model.eval(); result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(cfg["eval_batches"]):
            x, y = get_batch(data, cfg["batch_size"], cfg["block_size"], device)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite {name} evaluation loss.")
            losses.append(float(loss.item()))
        result[name] = float(np.mean(losses))
    model.train(was_training)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--session-steps", type=int, default=None,
                        help="Pause safely after this many optimizer steps; resume with the same config.")
    parser.add_argument("--dataset-manifest", required=True,
                        help="Dataset manifest; training_allowed must be explicitly true.")
    parser.add_argument("--source-manifest", required=True,
                        help="Manifest listing every approved training source.")
    parser.add_argument("--source-root", required=True,
                        help="Directory containing the files listed in the source manifest.")
    parser.add_argument("--corpus-report", required=True,
                        help="Audit report produced by prepare_data for the exact training corpus.")
    parser.add_argument("--source-review", default=None,
                        help="Evidence-backed per-domain approval CSV for URL-attributed records.")
    args = parser.parse_args()
    if args.session_steps is not None and args.session_steps < 1:
        parser.error("--session-steps must be a positive integer")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required = (
        "seed", "data_path", "tokenizer_path", "output_dir", "vocab_size",
        "block_size", "n_layer", "n_head", "n_embd", "dropout", "batch_size",
        "gradient_accumulation_steps", "learning_rate", "min_learning_rate",
        "weight_decay", "max_steps", "warmup_steps", "eval_interval",
        "eval_batches", "save_interval", "grad_clip", "amp",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing training config keys: {', '.join(missing)}")
    validate_base_training_config(cfg)
    tokenizer_path, data_path = Path(cfg["tokenizer_path"]), Path(cfg["data_path"])
    split_info, tokenizer_provenance_hash = validate_tokenizer_provenance(
        tokenizer_path,
        data_path,
        Path(args.dataset_manifest),
        Path(args.source_manifest),
        Path(args.source_root),
        Path(args.corpus_report),
        Path(args.source_review) if args.source_review else None,
    )
    print("Training approval gate passed: reviewed sources, reconstructed group-disjoint splits, tokenizer and provenance verified.")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != cfg["vocab_size"]:
        cfg["vocab_size"] = tokenizer.vocab_size
        print(f"Tokenizer vocabulary overrides config: {cfg['vocab_size']}")
    minimum = int(cfg["block_size"]) + 2
    train_data = load_tokens(str(split_info["train"]["path"]), tokenizer, min_tokens=minimum)
    val_data = load_tokens(str(split_info["validation"]["path"]), tokenizer, min_tokens=minimum)
    test_data = load_tokens(str(split_info["test"]["path"]), tokenizer, min_tokens=minimum)
    model_cfg = ModelConfig(**{k: cfg[k] for k in ("vocab_size","block_size","n_layer","n_head","n_embd","dropout")})
    model = AETHRIONLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    amp_enabled = device.type == "cuda" and bool(cfg["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    data_hash, tokenizer_hash = sha256_file(data_path), sha256_file(tokenizer_path)
    split_hashes = {name: split_info[name]["sha256"] for name in ("train", "validation", "test")}
    start_step = 0
    schedule_max_steps = cfg["max_steps"]
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        if checkpoint.get("model_config") != model.config_dict(): raise ValueError("Checkpoint model config differs.")
        if checkpoint.get("data_sha256") != data_hash: raise ValueError("Training corpus changed; refusing unsafe resume.")
        if checkpoint.get("tokenizer_sha256") != tokenizer_hash: raise ValueError("Tokenizer changed; refusing unsafe resume.")
        if checkpoint.get("tokenizer_provenance_sha256") != tokenizer_provenance_hash:
            raise ValueError("Tokenizer provenance changed; refusing unsafe resume.")
        if checkpoint.get("split_sha256") != split_hashes:
            raise ValueError("Approved train/validation/test split changed; refusing unsafe resume.")
        saved_training_config = checkpoint.get("training_config")
        current_training_config = _base_resume_training_config(cfg, cfg["max_steps"])
        if (
            not isinstance(saved_training_config, dict)
            or set(saved_training_config) != set(current_training_config)
            or any(type(saved_training_config[key]) is not type(value)
                   for key, value in current_training_config.items())
        ):
            raise ValueError("Resume checkpoint is missing or has an invalid BASE training configuration.")
        ignored_horizon = {"max_steps", "schedule_max_steps"}
        saved_fixed = {key: value for key, value in saved_training_config.items() if key not in ignored_horizon}
        current_fixed = {key: value for key, value in current_training_config.items() if key not in ignored_horizon}
        if saved_fixed != current_fixed:
            raise ValueError("BASE training settings changed; refusing a non-reproducible resume.")
        if cfg["max_steps"] < saved_training_config["max_steps"]:
            raise ValueError("Resume max_steps cannot be lower than the checkpoint training horizon.")
        schedule_max_steps = saved_training_config["schedule_max_steps"]
        if (
            schedule_max_steps < cfg["warmup_steps"]
            or schedule_max_steps > saved_training_config["max_steps"]
        ):
            raise ValueError("Resume checkpoint learning-rate schedule horizon is invalid.")
        checkpoint_step = checkpoint.get("step")
        if type(checkpoint_step) is not int or checkpoint_step < 0:
            raise ValueError("Checkpoint step is invalid.")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint: scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint_step + 1
        if start_step >= cfg["max_steps"]:
            raise ValueError("max_steps must be greater than the verified resume checkpoint step.")
        if "rng_state" in checkpoint: restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed from step {start_step}")
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    progress_path = output / "progress.json"
    started = time.perf_counter()
    last_metrics = {}
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(stop_signal, _request_stop)
    model.train()
    interrupted = False
    for step in range(start_step, cfg["max_steps"]):
        if step < cfg["warmup_steps"]:
            lr = cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
        else:
            progress = (step - cfg["warmup_steps"]) / max(1, schedule_max_steps - cfg["warmup_steps"])
            lr = cfg["min_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["gradient_accumulation_steps"]):
            x, y = get_batch(train_data, cfg["batch_size"], cfg["block_size"], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite BASE training loss at step {step}.")
                loss = loss / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["grad_clip"], error_if_nonfinite=True
        )
        if not torch.isfinite(grad_norm).item():
            raise FloatingPointError(f"Non-finite BASE gradient norm at step {step}.")
        scaler.step(optimizer); scaler.update()
        session_limit_reached = (
            args.session_steps is not None
            and (step - start_step + 1) >= args.session_steps
            and step < cfg["max_steps"] - 1
        )
        if step % cfg["eval_interval"] == 0 or session_limit_reached:
            last_metrics = estimate_loss(model, train_data, val_data, cfg, device)
            elapsed = max(time.perf_counter() - started, 1e-9)
            tokens_processed = (step + 1) * cfg["batch_size"] * cfg["block_size"] * cfg["gradient_accumulation_steps"]
            record = {
                "step": step,
                "train_loss": last_metrics["train"],
                "val_loss": last_metrics["val"],
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "tokens_processed": tokens_processed,
                "tokens_per_second": tokens_processed / elapsed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                import os
                os.fsync(handle.fileno())
            write_json_atomic({"status": "running", "max_steps": cfg["max_steps"], **record}, progress_path)
            print(f"step {step}: train={last_metrics['train']:.4f} val={last_metrics['val']:.4f} lr={lr:.2e}")

        should_save = (
            step % cfg["save_interval"] == 0
            or step == cfg["max_steps"] - 1
            or _STOP_REQUESTED
            or session_limit_reached
        )
        if should_save:
            payload = {
                "format_version": 4,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model_config": model.config_dict(),
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_hash,
                "tokenizer_provenance_sha256": tokenizer_provenance_hash,
                "data_path": str(data_path),
                "data_sha256": data_hash,
                "split_sha256": split_hashes,
                "rng_state": capture_rng_state(),
                "training_config": _base_resume_training_config(cfg, schedule_max_steps),
            }
            target = output / "checkpoint_last.pt"
            checksum = save_checkpoint(payload, target)
            print(f"Checkpoint saved at step {step} | sha256={checksum}", flush=True)

        if _STOP_REQUESTED or session_limit_reached:
            elapsed = max(time.perf_counter() - started, 1e-9)
            write_json_atomic({
                "status": "interrupted",
                "step": step,
                "max_steps": cfg["max_steps"],
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **last_metrics,
            }, progress_path)
            print("Training safely paused after checkpoint save. Resume with --resume.", flush=True)
            interrupted = True
            break

    if not interrupted:
        elapsed = max(time.perf_counter() - started, 1e-9)
        final_step = max(start_step - 1, cfg["max_steps"] - 1)
        test_metrics = estimate_loss(model, train_data, test_data, cfg, device)
        write_json_atomic({
            "status": "completed",
            "step": final_step,
            "max_steps": cfg["max_steps"],
            "elapsed_seconds": elapsed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **last_metrics,
            "held_out_test_loss": test_metrics["val"],
        }, progress_path)
        checkpoint_path = output / "checkpoint_last.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError("Completed BASE training has no final checkpoint.")
        write_json_atomic({
            "status": "training_complete_not_released",
            "release_status": "NOT_RELEASED",
            "model_id": "chat",
            "step": final_step,
            "dataset_sha256": data_hash,
            "split_sha256": split_hashes,
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_provenance_sha256": tokenizer_provenance_hash,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_config": model.config_dict(),
            "metrics": last_metrics,
            "held_out_test_loss": test_metrics["val"],
            "device": str(device),
            "note": "Independent instruction-following, factuality, multilingual and safety benchmarks are mandatory before release."
        }, output / "training_report.json")
        print("Training finished. Evaluate instruction-following and safety on independent held-out datasets before release.")

def validate_base_training_config(cfg: dict) -> None:
    """Reject malformed, extreme or non-finite BASE training settings."""
    if not isinstance(cfg, dict):
        raise ValueError("BASE training config must be a JSON object.")
    integer_fields = {
        "vocab_size": (256, 262144),
        "block_size": (8, 16384),
        "n_layer": (1, 256),
        "n_head": (1, 256),
        "n_embd": (8, 16384),
        "batch_size": (1, 4096),
        "gradient_accumulation_steps": (1, 4096),
        "max_steps": (1, 10_000_000),
        "warmup_steps": (0, 10_000_000),
        "eval_interval": (1, 10_000_000),
        "eval_batches": (1, 10_000_000),
        "save_interval": (1, 10_000_000),
    }
    for name, (minimum, maximum) in integer_fields.items():
        require_int(cfg.get(name), name, minimum=minimum, maximum=maximum)
    require_int(cfg.get("seed"), "seed", minimum=0, maximum=2**32 - 1)
    require_finite_number(cfg.get("learning_rate"), "learning_rate", minimum=0, minimum_inclusive=False)
    require_finite_number(cfg.get("min_learning_rate"), "min_learning_rate", minimum=0)
    require_finite_number(cfg.get("weight_decay"), "weight_decay", minimum=0)
    require_finite_number(cfg.get("grad_clip"), "grad_clip", minimum=0, minimum_inclusive=False)
    require_finite_number(cfg.get("dropout"), "dropout", minimum=0, maximum=1, maximum_inclusive=False)
    if cfg["min_learning_rate"] > cfg["learning_rate"]:
        raise ValueError("min_learning_rate must not exceed learning_rate.")
    if cfg["warmup_steps"] > cfg["max_steps"]:
        raise ValueError("warmup_steps must not exceed max_steps.")
    if type(cfg.get("amp")) is not bool:
        raise ValueError("amp must be a JSON boolean.")
    for name in ("data_path", "tokenizer_path", "output_dir"):
        if not isinstance(cfg.get(name), str) or not cfg[name].strip():
            raise ValueError(f"{name} must be a non-empty path string.")


if __name__ == "__main__": main()
