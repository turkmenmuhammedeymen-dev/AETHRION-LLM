"""Training loop with strict data checks and reproducible resumable checkpoints."""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from aethrion.model import AETHRIONLM, ModelConfig
from aethrion.tokenizer import ByteBPETokenizer

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state: torch.cuda.set_rng_state_all(state["cuda"])

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def load_tokens(path: str, tokenizer: ByteBPETokenizer) -> torch.Tensor:
    ids = tokenizer.encode(Path(path).read_text(encoding="utf-8"))
    if len(ids) < 100: raise ValueError("Corpus is too small; provide at least 100 tokens of licensed text.")
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
            _, loss = model(x, y); losses.append(loss.item())
        result[name] = float(np.mean(losses))
    model.train(was_training)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required = ("seed","data_path","tokenizer_path","output_dir","vocab_size","block_size","n_layer","n_head",
                "n_embd","dropout","batch_size","gradient_accumulation_steps","learning_rate",
                "min_learning_rate","weight_decay","max_steps","warmup_steps","eval_interval",
                "eval_batches","save_interval","train_split","grad_clip","amp")
    missing = [key for key in required if key not in cfg]
    if missing: raise ValueError(f"Missing training config keys: {', '.join(missing)}")
    if not 0.0 < cfg["train_split"] < 1.0: raise ValueError("train_split must be strictly between 0 and 1")
    for key in ("batch_size","gradient_accumulation_steps","max_steps","eval_interval","eval_batches","save_interval"):
        if cfg[key] < 1: raise ValueError(f"{key} must be >= 1")
    if cfg["warmup_steps"] < 0 or cfg["learning_rate"] <= 0 or cfg["min_learning_rate"] < 0:
        raise ValueError("Invalid learning-rate or warmup configuration")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer_path, data_path = Path(cfg["tokenizer_path"]), Path(cfg["data_path"])
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != cfg["vocab_size"]:
        cfg["vocab_size"] = tokenizer.vocab_size
        print(f"Tokenizer vocabulary overrides config: {cfg['vocab_size']}")
    tokens = load_tokens(str(data_path), tokenizer)
    split = int(len(tokens) * cfg["train_split"]); minimum = cfg["block_size"] + 2
    if split < minimum or len(tokens) - split < minimum:
        raise ValueError(f"Corpus needs at least {minimum} tokens in both train and validation splits; add licensed text or adjust block_size/train_split.")
    train_data, val_data = tokens[:split], tokens[split:]
    model_cfg = ModelConfig(**{k: cfg[k] for k in ("vocab_size","block_size","n_layer","n_head","n_embd","dropout")})
    model = AETHRIONLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    amp_enabled = device.type == "cuda" and bool(cfg["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    data_hash, tokenizer_hash = sha256_file(data_path), sha256_file(tokenizer_path)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != model.config_dict(): raise ValueError("Checkpoint model config differs.")
        if checkpoint.get("data_sha256") != data_hash: raise ValueError("Training corpus changed; refusing unsafe resume.")
        if checkpoint.get("tokenizer_sha256") != tokenizer_hash: raise ValueError("Tokenizer changed; refusing unsafe resume.")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint: scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"]) + 1
        if "rng_state" in checkpoint: restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed from step {start_step}")
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        if step < cfg["warmup_steps"]:
            lr = cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
        else:
            progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
            lr = cfg["min_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["gradient_accumulation_steps"]):
            x, y = get_batch(train_data, cfg["batch_size"], cfg["block_size"], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                _, loss = model(x, y); loss = loss / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer); scaler.update()
        if step % cfg["eval_interval"] == 0:
            metrics = estimate_loss(model, train_data, val_data, cfg, device)
            print(f"step {step}: train={metrics['train']:.4f} val={metrics['val']:.4f} lr={lr:.2e}")
        if step % cfg["save_interval"] == 0 or step == cfg["max_steps"] - 1:
            payload = {"format_version":2,"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                       "scaler":scaler.state_dict(),"model_config":model.config_dict(),
                       "tokenizer_path":str(tokenizer_path),"tokenizer_sha256":tokenizer_hash,
                       "data_path":str(data_path),"data_sha256":data_hash,"rng_state":capture_rng_state()}
            target = output / "checkpoint_last.pt"; temporary = output / "checkpoint_last.pt.tmp"
            torch.save(payload, temporary); temporary.replace(target)
            print(f"Checkpoint saved at step {step}")
    print("Training finished. Evaluate on an independent held-out corpus before capability claims.")

if __name__ == "__main__": main()
