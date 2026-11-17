"""Shared, resumable training runner for first-party specialist models.

Training is impossible until specialist_data.py verifies the explicit dataset,
source-license, privacy and consent approvals. No model is automatically
marked trained or released by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aethrion.specialist_data import validate_specialist_dataset
from aethrion.checkpoint import write_json_atomic


_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Stop requested (signal {signum}); checkpointing after the current step.", flush=True)


def _build_model(model_id: str, config: dict[str, Any]) -> nn.Module:
    if model_id == "code":
        from aethrion.code_model import AETHRIONCodeLM, CodeModelConfig
        return AETHRIONCodeLM(CodeModelConfig(**config))
    if model_id == "vision":
        from aethrion.vision_model import AETHRIONVision, VisionConfig
        return AETHRIONVision(VisionConfig(**config))
    if model_id == "image":
        from aethrion.image_model import AETHRIONImageGenerator, ImageModelConfig
        return AETHRIONImageGenerator(ImageModelConfig(**config))
    if model_id == "asr":
        from aethrion.asr_model import AETHRIONASR, ASRConfig
        return AETHRIONASR(ASRConfig(**config))
    if model_id == "tts":
        from aethrion.tts_model import AETHRIONTTS, TTSConfig
        return AETHRIONTTS(TTSConfig(**config))
    if model_id == "vocoder":
        from aethrion.tts_model import AETHRIONVocoder
        return AETHRIONVocoder(**config)
    if model_id == "3d":
        from aethrion.model_3d import AETHRION3D, Model3DConfig
        return AETHRION3D(Model3DConfig(**config))
    if model_id == "research":
        from aethrion.research_model import AETHRIONResearch, ResearchConfig
        return AETHRIONResearch(ResearchConfig(**config))
    if model_id == "document":
        from aethrion.document_model import AETHRIONDocument, DocumentConfig
        return AETHRIONDocument(DocumentConfig(**config))
    if model_id == "safety":
        from aethrion.safety_model import AETHRIONSafe, SafetyConfig
        return AETHRIONSafe(SafetyConfig(**config))
    raise ValueError(f"Unsupported specialist model_id: {model_id}")


def _to_batch(arrays: dict[str, np.ndarray], keys: list[str], indices: torch.Tensor,
              device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    if isinstance(indices, torch.Tensor):
        index_np = indices.detach().cpu().numpy()
    else:
        index_np = np.asarray(indices, dtype=np.int64)
    for key in keys:
        value = np.asarray(arrays[key][index_np])
        tensor = torch.from_numpy(value.copy())
        if key in {
            "token_ids", "query_ids", "passage_ids", "layout_ids", "tag_labels",
            "decision_labels", "reason_labels", "input_lengths", "target_lengths", "targets",
            "frame_to_token", "start_positions", "end_positions",
        }:
            tensor = tensor.long()
        elif tensor.dtype == torch.uint8 and key in {"images"}:
            tensor = tensor.float().div(127.5).sub(1.0)
        elif tensor.dtype in (torch.float64, torch.float16, torch.bfloat16):
            tensor = tensor.float()
        batch[key] = tensor.to(device)
    return batch


def _loss(model: nn.Module, model_id: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_id == "code":
        ids = batch["token_ids"]
        if ids.ndim != 2 or ids.size(1) < 2:
            raise ValueError("code token_ids must have shape [batch, sequence>=2]")
        return model(ids[:, :-1], ids[:, 1:])[1]
    if model_id == "vision":
        return model(batch["images"], batch["token_ids"], batch.get("attention_mask"))["loss"]
    if model_id == "image":
        return model.diffusion_loss(batch["images"], batch["token_ids"])
    if model_id == "asr":
        targets = batch["targets"]
        lengths = batch["target_lengths"]
        if targets.ndim != 2 or lengths.ndim != 1:
            raise ValueError("ASR targets must be padded [batch,target_length] with target_lengths")
        flattened = torch.cat([targets[i, :int(lengths[i].item())] for i in range(targets.size(0))])
        return model.ctc_loss(batch["features"], flattened, batch["input_lengths"], lengths)
    if model_id == "tts":
        acoustic = model.acoustic_loss(
            batch["token_ids"], batch["frame_to_token"], batch["target_mels"], batch.get("frame_mask")
        )
        duration = model.duration_loss(batch["token_ids"], batch["durations"], batch.get("token_mask"))
        return acoustic + 0.1 * duration
    if model_id == "vocoder":
        predicted = model(batch["target_mels"].transpose(1, 2))
        return model.waveform_loss(predicted, batch["target_waveforms"], batch.get("waveform_mask"))
    if model_id == "3d":
        return model.diffusion_loss(batch["points"], batch["token_ids"])
    if model_id == "research":
        return model(
            batch["query_ids"], batch["passage_ids"],
            batch.get("query_mask"), batch.get("passage_mask")
        )["loss"]
    if model_id == "document":
        output = model(batch["token_ids"], batch["boxes"], batch["layout_ids"], batch.get("attention_mask"))
        return model.multitask_loss(
            output, tag_labels=batch.get("tag_labels"),
            start_positions=batch.get("start_positions"), end_positions=batch.get("end_positions")
        )
    if model_id == "safety":
        output = model(batch["token_ids"], batch.get("attention_mask"))
        return model.loss(output, batch["decision_labels"], batch.get("reason_labels"))
    raise ValueError(f"No training objective registered for {model_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    py = random.getstate()
    np_state = np.random.get_state()
    state = {
        "python": {"version": int(py[0]), "state": list(py[1]), "gauss": py[2]},
        "numpy": {
            "algorithm": str(np_state[0]), "keys": np_state[1].tolist(),
            "position": int(np_state[2]), "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    py = state["python"]
    random.setstate((int(py["version"]), tuple(int(x) for x in py["state"]), py["gauss"]))
    ns = state["numpy"]
    np.random.set_state((
        str(ns["algorithm"]), np.asarray(ns["keys"], dtype=np.uint32),
        int(ns["position"]), int(ns["has_gauss"]), float(ns["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    """Atomically save a checkpoint and retain the previous verified generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    checksum_path = Path(str(path) + ".sha256")
    checksum_tmp = Path(str(checksum_path) + ".tmp")
    previous = Path(str(path) + ".previous")
    previous_checksum = Path(str(previous) + ".sha256")
    previous_tmp = Path(str(previous) + ".tmp")
    previous_checksum_tmp = Path(str(previous_checksum) + ".tmp")
    try:
        torch.save(payload, tmp)
        with tmp.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(tmp)
        checksum_tmp.write_text(digest + "\n", encoding="ascii")
        if path.is_file() and checksum_path.is_file():
            expected = checksum_path.read_text(encoding="ascii").strip().split()
            if expected and _sha256(path) == expected[0]:
                shutil.copy2(path, previous_tmp)
                shutil.copy2(checksum_path, previous_checksum_tmp)
                os.replace(previous_tmp, previous)
                os.replace(previous_checksum_tmp, previous_checksum)
        os.replace(tmp, path)
        os.replace(checksum_tmp, checksum_path)
        return digest
    finally:
        for candidate in (tmp, checksum_tmp, previous_tmp, previous_checksum_tmp):
            candidate.unlink(missing_ok=True)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    def verified(candidate: Path) -> bool:
        sidecar = Path(str(candidate) + ".sha256")
        if not candidate.is_file() or not sidecar.is_file():
            return False
        expected = sidecar.read_text(encoding="ascii").strip().split()
        return bool(expected and _sha256(candidate) == expected[0])

    load_path = path
    if not verified(path):
        previous = Path(str(path) + ".previous")
        if not verified(previous):
            raise ValueError("Resume checkpoint and its verified previous generation are unavailable.")
        load_path = previous
    payload = torch.load(load_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "aethrion-specialist-checkpoint-v1":
        raise ValueError("Unsupported specialist checkpoint format.")
    required = {"model_id", "step", "model_config", "model_state", "optimizer_state",
                "dataset_sha256", "rng_state"}
    if required - set(payload):
        raise ValueError("Specialist checkpoint is missing required resume fields.")
    return payload


def _evaluate(model, model_id, arrays, keys, indices, device, batch_size):
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start:start + batch_size]
            batch = _to_batch(arrays, keys, batch_ids, device)
            loss = _loss(model, model_id, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {model_id} evaluation loss.")
            losses.append(float(loss.item()))
    model.train(was_training)
    if not losses:
        raise ValueError("Evaluation split is empty.")
    return float(sum(losses) / len(losses))


def train_specialist(
    config: dict[str, Any],
    manifest_path: str | Path,
    *,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train one independent specialist using a licensed, approved local NPZ dataset."""
    if not isinstance(config, dict):
        raise TypeError("Training config must be a JSON object.")
    required = {"model_id", "model_config", "output_dir", "seed", "max_steps",
                "batch_size", "learning_rate", "weight_decay", "grad_clip",
                "eval_interval", "save_interval"}
    missing = required - set(config)
    if missing:
        raise ValueError("Missing specialist training keys: " + ", ".join(sorted(missing)))
    for key in ("max_steps", "batch_size", "eval_interval", "save_interval"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer configuration.")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    approval = validate_specialist_dataset(manifest_path, require_training_approval=True)
    model_id = config["model_id"]
    if approval["model_id"] != model_id:
        raise ValueError("Training config model_id differs from approved dataset manifest.")

    manifest_file = Path(manifest_path).resolve()
    with np.load(approval["dataset_path"], allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    split_names = arrays["split"].astype(str)
    train_idx = np.flatnonzero(split_names == "train")
    val_idx = np.flatnonzero(split_names == "validation")
    test_idx = np.flatnonzero(split_names == "test")
    keys = sorted(set(arrays) - {"sample_id", "split"})
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("Train, validation and test partitions must all be populated.")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_id, config["model_config"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_last.pt"
    start_step = 0
    if resume is not None:
        state = _load_checkpoint(Path(resume), device)
        if state.get("model_id") != model_id or state.get("dataset_sha256") != approval["dataset_sha256"]:
            raise ValueError("Resume model or approved dataset differs from checkpoint.")
        if state.get("model_config") != config["model_config"]:
            raise ValueError("Resume model configuration differs from checkpoint.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _restore_rng_state(state["rng_state"])
        checkpoint_step = state.get("step")
        if type(checkpoint_step) is not int or checkpoint_step < 0:
            raise ValueError("Resume checkpoint step is invalid.")
        start_step = checkpoint_step + 1
        if start_step >= config["max_steps"]:
            raise ValueError("max_steps must be greater than the verified resume checkpoint step.")

    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    last_validation = None
    model.train()
    for step in range(start_step, config["max_steps"]):
        indices = torch.randint(len(train_idx), (config["batch_size"],))
        batch_indices = torch.as_tensor(train_idx, dtype=torch.long)[indices]
        batch = _to_batch(arrays, keys, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, model_id, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {model_id} training loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()

        if step % config["eval_interval"] == 0 or step == config["max_steps"] - 1 or _STOP_REQUESTED:
            last_validation = _evaluate(
                model, model_id, arrays, keys, val_idx, device, config["batch_size"]
            )
            record = {
                "model_id": model_id, "step": step, "train_loss": float(loss.item()),
                "validation_loss": last_validation, "seconds": time.perf_counter() - started,
                "dataset_sha256": approval["dataset_sha256"],
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(record, ensure_ascii=False), flush=True)

        if step % config["save_interval"] == 0 or step == config["max_steps"] - 1 or _STOP_REQUESTED:
            checkpoint_hash = _save_checkpoint(checkpoint_path, {
                "format": "aethrion-specialist-checkpoint-v1",
                "model_id": model_id, "step": step,
                "model_config": config["model_config"],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "dataset_sha256": approval["dataset_sha256"],
                "rng_state": _capture_rng_state(),
            })
            write_json_atomic({
                "status": "training_in_progress_not_released",
                "release_status": "NOT_RELEASED",
                "model_id": model_id, "step": step,
                "dataset_sha256": approval["dataset_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "validation_loss": last_validation,
                "note": "Training alone does not authorize registry status=trained; run independent quality and safety evaluation."
            }, output / "training_report.json")

        if _STOP_REQUESTED and step < config["max_steps"] - 1:
            interrupted_report = {
                "status": "training_interrupted_not_released",
                "release_status": "NOT_RELEASED",
                "model_id": model_id,
                "steps": step + 1,
                "dataset_sha256": approval["dataset_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "validation_loss": last_validation,
                "sample_counts": approval["split_counts"],
                "device": str(device),
                "note": "Training was safely paused after a verified checkpoint; resume with the same dataset and model configuration."
            }
            write_json_atomic(interrupted_report, output / "training_report.json")
            return interrupted_report

    test_loss = _evaluate(model, model_id, arrays, keys, test_idx, device, config["batch_size"])
    report = {
        "status": "training_complete_not_released",
        "release_status": "NOT_RELEASED",
        "model_id": model_id,
        "steps": config["max_steps"],
        "dataset_sha256": approval["dataset_sha256"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_loss": last_validation,
        "held_out_test_loss": test_loss,
        "sample_counts": approval["split_counts"],
        "device": str(device),
        "note": "Independent benchmark, calibration, safety review and release approval remain required."
    }
    write_json_atomic(report, output / "training_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train one AETHRION specialist model from scratch.")
    parser.add_argument("--config", required=True, help="JSON model and optimizer configuration.")
    parser.add_argument("--manifest", required=True, help="Approved specialist dataset manifest.")
    parser.add_argument("--resume", default=None, help="Optional SHA-256 verified checkpoint.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous_handlers = {}
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[stop_signal] = signal.signal(stop_signal, _request_stop)
    try:
        report = train_specialist(config, args.manifest, resume=args.resume)
    finally:
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
