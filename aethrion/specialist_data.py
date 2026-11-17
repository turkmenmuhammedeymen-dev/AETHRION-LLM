"""Fail-closed governance and integrity checks for specialist multimodal datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

FORMAT = "aethrion-specialist-dataset-v1"
MODEL_MODALITIES = {
    "chat": "text", "code": "code", "vision": "vision", "image": "image", "asr": "audio",
    "tts": "audio", "vocoder": "audio", "3d": "3d", "research": "text",
    "document": "document", "safety": "text",
}
REQUIRED_ARRAYS = {
    "chat": {"sample_id", "split", "input_ids", "labels"},
    "code": {"sample_id", "split", "token_ids"},
    "vision": {"sample_id", "split", "images", "token_ids"},
    "image": {"sample_id", "split", "images", "token_ids"},
    "asr": {"sample_id", "split", "features", "input_lengths", "targets", "target_lengths"},
    "tts": {"sample_id", "split", "token_ids", "frame_to_token", "target_mels", "durations"},\n    "vocoder": {"sample_id", "split", "target_mels", "target_waveforms"},
    "3d": {"sample_id", "split", "points", "token_ids"},
    "research": {"sample_id", "split", "query_ids", "passage_ids"},
    "document": {"sample_id", "split", "token_ids", "boxes", "layout_ids", "tag_labels"},
    "safety": {"sample_id", "split", "token_ids", "decision_labels", "reason_labels"},
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SpecialistDataError(ValueError):
    """Invalid, unapproved, or integrity-failing specialist dataset."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_local_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or "\\" in relative or "://" in relative:
        raise SpecialistDataError("Dataset paths must be relative local paths.")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise SpecialistDataError("Dataset path contains an unsafe component.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SpecialistDataError("Dataset path escapes the declared data root.") from exc
    return resolved


def validate_specialist_dataset(
    manifest_path: str | Path,
    *,
    require_training_approval: bool = True,
) -> dict[str, Any]:
    """Validate license evidence, exact approval booleans, NPZ schema, splits and hashes."""
    manifest_file = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpecialistDataError("Specialist dataset manifest is missing or invalid JSON.") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise SpecialistDataError("Unsupported specialist dataset manifest format.")
    model_id = payload.get("model_id")
    if model_id not in MODEL_MODALITIES:
        raise SpecialistDataError("Unknown specialist model_id.")
    if payload.get("modality") != MODEL_MODALITIES[model_id]:
        raise SpecialistDataError("Dataset modality does not match the target specialist.")
    if type(payload.get("training_allowed")) is not bool:
        raise SpecialistDataError("training_allowed must be an exact JSON boolean.")
    if type(payload.get("owner_approval")) is not bool:
        raise SpecialistDataError("owner_approval must be an exact JSON boolean.")
    if require_training_approval and (
        payload["training_allowed"] is not True or payload["owner_approval"] is not True
    ):
        raise SpecialistDataError("Training blocked: specialist dataset and owner approvals are not true.")
    if payload.get("privacy_reviewed") is not True:
        raise SpecialistDataError("Training blocked: privacy review is not approved.")
    if model_id in {"asr", "tts", "vocoder"} and payload.get("speaker_consent_reviewed") is not True:
        raise SpecialistDataError("Training blocked: speaker consent review is not approved.")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SpecialistDataError("At least one documented training source is required.")
    for source in sources:
        if not isinstance(source, dict):
            raise SpecialistDataError("Each source review must be an object.")
        required_text = ("source_id", "source_uri", "source_path", "license", "permitted_use", "evidence_uri")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in required_text):
            raise SpecialistDataError("Each source requires an ID, URI, license, permitted use and evidence.")
        digest = source.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest.lower()):
            raise SpecialistDataError("Each source requires a valid SHA-256.")
        source_file = _safe_local_path(manifest_file.parent.resolve(), source["source_path"])
        if not source_file.is_file() or source_file.stat().st_size == 0:
            raise SpecialistDataError("A reviewed source file is missing or empty.")
        if sha256_file(source_file) != digest.lower():
            raise SpecialistDataError("A reviewed source file SHA-256 mismatch.")
        if type(source.get("allow_training")) is not bool or source["allow_training"] is not True:
            raise SpecialistDataError("Training blocked: one or more source licenses are not approved.")
        if source.get("privacy_reviewed") is not True:
            raise SpecialistDataError("Training blocked: one or more sources lack privacy approval.")
        if model_id in {"asr", "tts"} and source.get("speaker_consent_verified") is not True:
            raise SpecialistDataError("Training blocked: audio source lacks verified speaker consent.")

    dataset_path = payload.get("dataset_path")
    dataset_hash = payload.get("dataset_sha256")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise SpecialistDataError("dataset_path is required.")
    if not isinstance(dataset_hash, str) or not SHA256.fullmatch(dataset_hash.lower()):
        raise SpecialistDataError("dataset_sha256 must be a SHA-256 digest.")
    root = manifest_file.parent.resolve()
    file_path = _safe_local_path(root, dataset_path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        raise SpecialistDataError("Prepared specialist dataset is missing or empty.")
    actual_hash = sha256_file(file_path)
    if actual_hash != dataset_hash.lower():
        raise SpecialistDataError("Prepared specialist dataset SHA-256 mismatch.")

def _validate_modality_shapes(model_id: str, arrays: Any, sample_count: int) -> None:
    """Validate fixed-shape tensors before any model or optimizer is constructed."""
    if model_id in {"chat", "code"}:
        key = "input_ids" if model_id == "chat" else "token_ids"
        ids = arrays[key]
        if ids.ndim != 2 or ids.shape[1] < 2 or not np.issubdtype(ids.dtype, np.integer):
            raise SpecialistDataError(f"{key} must be an integer [samples, sequence>=2] array.")
        if model_id == "chat":
            labels = arrays["labels"]
            if labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer):
                raise SpecialistDataError("Instruction labels must be an integer array matching input_ids.")
            if np.any(np.all(labels == -100, axis=1)):
                raise SpecialistDataError("Every instruction sample must contain supervised response tokens.")
    elif model_id in {"vision", "image"}:
        images, ids = arrays["images"], arrays["token_ids"]
        if images.ndim != 4 or images.shape[1] != 3 or ids.ndim != 2:
            raise SpecialistDataError("Image-model arrays require images [N,3,H,W] and token_ids [N,L].")
        if images.shape[-1] < 16 or images.shape[-2] < 16:
            raise SpecialistDataError("Image dimensions must be at least 16x16.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise SpecialistDataError("Image conditioning token IDs must be integers.")
    elif model_id == "asr":
        features, input_lengths = arrays["features"], arrays["input_lengths"]
        targets, target_lengths = arrays["targets"], arrays["target_lengths"]
        if features.ndim != 3 or input_lengths.shape != (sample_count,):
            raise SpecialistDataError("ASR features must be [N,frames,mels] with one input length per sample.")
        if targets.ndim != 2 or target_lengths.shape != (sample_count,):
            raise SpecialistDataError("ASR targets must be padded [N,target_length] with one target length per sample.")
        if np.any(input_lengths < 1) or np.any(input_lengths > features.shape[1]):
            raise SpecialistDataError("ASR input lengths are outside the feature sequence.")
        if np.any(target_lengths < 1) or np.any(target_lengths > targets.shape[1]):
            raise SpecialistDataError("ASR target lengths are outside the padded target sequence.")
    elif model_id == "tts":
        ids, alignment = arrays["token_ids"], arrays["frame_to_token"]
        mels, durations = arrays["target_mels"], arrays["durations"]
        if ids.ndim != 2 or alignment.ndim != 2 or mels.ndim != 3 or durations.shape != ids.shape:
            raise SpecialistDataError("TTS requires token_ids [N,L], frame_to_token [N,F], target_mels [N,F,M] and durations [N,L].")
        if alignment.shape[:2] != mels.shape[:2] or np.any(durations < 1):
            raise SpecialistDataError("TTS frame alignment or positive duration targets are invalid.")
        if np.any(durations.sum(axis=1) != alignment.shape[1]):
            raise SpecialistDataError("TTS token durations must sum to the target mel frame count.")
        if np.any(alignment < 0) or np.any(alignment >= ids.shape[1]):
            raise SpecialistDataError("TTS alignment references token positions outside the transcript.")
    elif model_id == "vocoder":
        mels, audio = arrays["target_mels"], arrays["target_waveforms"]
        if mels.ndim != 3 or audio.ndim != 2:
            raise SpecialistDataError("Vocoder requires target_mels [N,frames,mel_bins] and target_waveforms [N,samples].")
    elif model_id == "3d":
        points, ids = arrays["points"], arrays["token_ids"]
        if points.ndim != 3 or points.shape[2] != 3 or ids.ndim != 2:
            raise SpecialistDataError("3D data requires points [N,points,3] and token_ids [N,L].")
        if points.shape[1] < 16 or not np.issubdtype(ids.dtype, np.integer):
            raise SpecialistDataError("3D point clouds or conditioning token IDs are invalid.")
    elif model_id == "research":
        if arrays["query_ids"].ndim != 2 or arrays["passage_ids"].ndim != 2:
            raise SpecialistDataError("Research query_ids and passage_ids must be padded 2D integer arrays.")
        if not np.issubdtype(arrays["query_ids"].dtype, np.integer) or not np.issubdtype(arrays["passage_ids"].dtype, np.integer):
            raise SpecialistDataError("Research token arrays must use integer dtypes.")
    elif model_id == "document":
        ids, boxes, layouts, tags = arrays["token_ids"], arrays["boxes"], arrays["layout_ids"], arrays["tag_labels"]
        if ids.ndim != 2 or boxes.shape != (*ids.shape, 4) or layouts.shape != ids.shape or tags.shape != ids.shape:
            raise SpecialistDataError("Document token, layout, boxes and tag arrays have inconsistent shapes.")
        if np.any(boxes < 0) or np.any(boxes > 1):
            raise SpecialistDataError("Document boxes must be normalized to [0,1].")
    elif model_id == "safety":
        ids, decisions, reasons = arrays["token_ids"], arrays["decision_labels"], arrays["reason_labels"]
        if ids.ndim != 2 or decisions.shape != (sample_count,) or reasons.ndim != 2 or reasons.shape[0] != sample_count:
            raise SpecialistDataError("Safety token, decision and reason arrays have inconsistent shapes.")
        if np.any(decisions < 0) or np.any(decisions > 2):
            raise SpecialistDataError("Safety decision labels must be ALLOW=0, REVIEW=1 or BLOCK=2.")


    try:
        with np.load(file_path, allow_pickle=False) as arrays:
            keys = set(arrays.files)
            missing = REQUIRED_ARRAYS[model_id] - keys
            if missing:
                raise SpecialistDataError("Dataset is missing arrays: " + ", ".join(sorted(missing)))
            sample_ids = arrays["sample_id"]
            splits = arrays["split"]
            if sample_ids.ndim != 1 or splits.ndim != 1 or len(sample_ids) == 0 or len(sample_ids) != len(splits):
                raise SpecialistDataError("sample_id and split arrays must be non-empty 1D arrays of equal length.")
            ids = [str(item) for item in sample_ids.tolist()]
            split_names = [str(item) for item in splits.tolist()]
            if len(set(ids)) != len(ids) or any(not item for item in ids):
                raise SpecialistDataError("sample_id values must be unique and non-empty.")
            if set(split_names) - {"train", "validation", "test"}:
                raise SpecialistDataError("split values must be train, validation or test.")
            counts = {name: split_names.count(name) for name in ("train", "validation", "test")}
            if any(counts[name] == 0 for name in counts):
                raise SpecialistDataError("Dataset requires non-empty train, validation and test splits.")
            sample_count = len(ids)
            _validate_modality_shapes(model_id, arrays, sample_count)
            for key in keys - {"sample_id", "split"}:
                value = arrays[key]
                if value.dtype.hasobject:
                    raise SpecialistDataError(f"Object/pickle arrays are forbidden: {key}.")
                if value.ndim < 1 or value.shape[0] != sample_count:
                    raise SpecialistDataError(f"Array {key} must have {sample_count} samples in dimension zero.")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise SpecialistDataError(f"Array {key} contains non-finite values.")
    except SpecialistDataError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise SpecialistDataError("Prepared dataset is not a valid safe NPZ archive.") from exc

    return {
        "model_id": model_id,
        "modality": payload["modality"],
        "dataset_path": str(file_path),
        "dataset_sha256": actual_hash,
        "sample_count": sample_count,
        "split_counts": counts,
        "source_count": len(sources),
    }
