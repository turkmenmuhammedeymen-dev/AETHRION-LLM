"""Read-only semantic validation for native model artifacts and their provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "invalid_or_unreadable_json"
    if not isinstance(value, dict):
        return None, "json_root_not_object"
    return value, "ok"


def _model_for_validation(model_id: str, config: dict[str, Any]):
    if model_id == "chat":
        from aethrion.model import AETHRIONLM, ModelConfig
        return AETHRIONLM(ModelConfig(**config))
    from aethrion.specialist_train import _build_model
    return _build_model(model_id, config)


def _validate_checkpoint(path: Path, model_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_checked", "step": None, "tensor_count": 0}
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        result["status"] = "safe_load_failed"
        result["reason"] = type(exc).__name__
        return result
    if not isinstance(payload, dict):
        result["status"] = "invalid_checkpoint_root"
        return result

    if model_id == "chat":
        if payload.get("format_version") != 4:
            result["status"] = "unsupported_checkpoint_format"
            return result
        state = payload.get("model")
        step = payload.get("step")
    else:
        if payload.get("format") != "aethrion-specialist-checkpoint-v2":
            result["status"] = "unsupported_checkpoint_format"
            return result
        if payload.get("model_id") != model_id:
            result["status"] = "model_id_mismatch"
            return result
        state = payload.get("model_state")
        step = payload.get("step")

    if type(step) is not int or step < 0:
        result["status"] = "invalid_step"
        return result
    expected_config = entry.get("model_config")
    checkpoint_config = payload.get("model_config")
    if not isinstance(expected_config, dict) or checkpoint_config != expected_config:
        result["status"] = "model_config_mismatch"
        return result
    if not isinstance(state, dict) or not state:
        result["status"] = "missing_model_state"
        return result

    try:
        model = _model_for_validation(model_id, expected_config)
        expected_state = model.state_dict()
    except Exception as exc:
        result["status"] = "model_architecture_validation_failed"
        result["reason"] = type(exc).__name__
        return result

    if set(state) != set(expected_state):
        result["status"] = "weight_key_mismatch"
        result["missing_weight_keys"] = sorted(set(expected_state) - set(state))[:30]
        result["unexpected_weight_keys"] = sorted(set(state) - set(expected_state))[:30]
        return result

    for key, expected_tensor in expected_state.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            result["status"] = "non_tensor_weight"
            result["weight_key"] = key
            return result
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            result["status"] = "weight_shape_mismatch"
            result["weight_key"] = key
            return result
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            result["status"] = "non_finite_weight"
            result["weight_key"] = key
            return result

    dataset_hash = payload.get("data_sha256") if model_id == "chat" else payload.get("dataset_sha256")
    report = {
        "status": "verified_structure",
        "step": step,
        "tensor_count": len(state),
        "dataset_sha256": dataset_hash,
        "checkpoint_format": payload.get("format_version", payload.get("format")),
    }
    return report


def audit_model_artifact_semantics(root: str | Path, model_id: str,
                                   entry: dict[str, Any]) -> dict[str, Any]:
    """Validate reports, tokenizer/processor schemas, and actual checkpoint weights.

    This function is deliberately read-only. It never changes registry status or
    approval flags and never treats artifact presence as proof of training.
    """
    repo = Path(root).resolve()
    output: dict[str, Any] = {
        "model_id": model_id,
        "training_report_status": "missing",
        "evaluation_report_status": "missing",
        "tokenizer_status": "not_applicable" if model_id == "vocoder" else "missing",
        "processor_status": "not_required",
        "checkpoint": {"status": "missing", "tensor_count": 0, "step": None},
        "cross_artifact_consistent": False,
        "trained_claim_supported": False,
    }

    def local_file(key: str) -> Path | None:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = Path(value)
        if candidate.is_absolute() or chr(92) in value or "://" in value:
            return None
        if any(part in {"", ".", ".."} for part in value.split("/")):
            return None
        resolved = (repo / candidate).resolve()
        try:
            resolved.relative_to(repo)
        except ValueError:
            return None
        return resolved if resolved.is_file() else None

    training = None
    evaluation = None
    training_path = local_file("training_report")
    evaluation_path = local_file("evaluation_report")
    if training_path:
        training, status = _read_json(training_path)
        output["training_report_status"] = status
    if evaluation_path:
        evaluation, status = _read_json(evaluation_path)
        output["evaluation_report_status"] = status

    checkpoint_path = local_file("checkpoint")
    checkpoint_hash = None
    if checkpoint_path:
        checkpoint_hash = _digest(checkpoint_path)
        output["checkpoint"] = _validate_checkpoint(checkpoint_path, model_id, entry)
        output["checkpoint"]["sha256"] = checkpoint_hash
        registered_hash = entry.get("checkpoint_sha256")
        if registered_hash and registered_hash.lower() != checkpoint_hash:
            output["checkpoint"]["status"] = "registered_hash_mismatch"

    if model_id != "vocoder":
        tokenizer_path = local_file("tokenizer")
        if tokenizer_path:
            try:
                from aethrion.tokenizer import ByteBPETokenizer
                tokenizer = ByteBPETokenizer.load(tokenizer_path)
                configured_vocab = (entry.get("model_config") or {}).get("vocab_size")
                output["tokenizer_status"] = (
                    "verified" if configured_vocab is None or tokenizer.vocab_size <= configured_vocab
                    else "vocab_exceeds_model_config"
                )
                output["tokenizer_vocab_size"] = tokenizer.vocab_size
            except Exception as exc:
                output["tokenizer_status"] = "invalid_tokenizer"
                output["tokenizer_reason"] = type(exc).__name__

    processor_required = model_id in {
        "vision", "image", "asr", "tts", "3d", "research", "document", "safety"
    }
    if processor_required:
        output["processor_status"] = "missing"
        processor_path = local_file("processor")
        if processor_path:
            processor, processor_status = _read_json(processor_path)
            if processor is None:
                output["processor_status"] = processor_status
            elif processor.get("model_id", model_id) != model_id:
                output["processor_status"] = "model_id_mismatch"
            elif not isinstance(processor, dict) or not processor:
                output["processor_status"] = "invalid_processor"
            else:
                output["processor_status"] = "verified_structure"
                output["processor_format"] = processor.get("format")

    problems = []
    if training is not None:
        if training.get("model_id") != model_id:
            problems.append("training_report_model_id_mismatch")
        if not isinstance(training.get("dataset_sha256"), str):
            problems.append("training_report_dataset_hash_missing")
        if checkpoint_hash and training.get("checkpoint_sha256") != checkpoint_hash:
            problems.append("training_report_checkpoint_hash_mismatch")
        checkpoint_dataset_hash = output["checkpoint"].get("dataset_sha256")
        if checkpoint_dataset_hash and training.get("dataset_sha256") != checkpoint_dataset_hash:
            problems.append("training_report_dataset_hash_mismatch")
        checkpoint_step = output["checkpoint"].get("step")
        report_step = training.get("step", training.get("steps"))
        if checkpoint_step is not None and report_step is not None and checkpoint_step != report_step:
            problems.append("training_report_step_mismatch")
        if training.get("release_status") not in {None, "NOT_RELEASED"}:
            problems.append("unexpected_training_report_release_status")
        step = training.get("step", training.get("steps"))
        if step is not None and (type(step) is not int or step < 0):
            problems.append("invalid_training_step")
        for metric in ("validation_loss", "held_out_test_loss"):
            value = training.get(metric)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0
            ):
                problems.append(f"invalid_{metric}")

    if evaluation is not None:
        if evaluation.get("model_id") != model_id:
            problems.append("evaluation_report_model_id_mismatch")
        if training and evaluation.get("dataset_sha256") != training.get("dataset_sha256"):
            problems.append("evaluation_dataset_hash_mismatch")
        if checkpoint_hash and evaluation.get("checkpoint_sha256") != checkpoint_hash:
            problems.append("evaluation_checkpoint_hash_mismatch")

    if training is not None and checkpoint_hash:
        tokenizer_ok = model_id == "vocoder" or output["tokenizer_status"] == "verified"
        processor_ok = not processor_required or output["processor_status"] == "verified_structure"
        output["cross_artifact_consistent"] = (
            not problems
            and output["checkpoint"]["status"] == "verified_structure"
            and tokenizer_ok
            and processor_ok
            and training.get("model_id") == model_id
            and training.get("checkpoint_sha256") == checkpoint_hash
        )
    output["consistency_problems"] = problems
    output["trained_claim_supported"] = False
    return output
