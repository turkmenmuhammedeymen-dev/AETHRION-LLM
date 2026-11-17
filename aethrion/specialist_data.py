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
    "tts": "audio", "3d": "3d", "research": "text",
    "document": "document", "safety": "text",
}
REQUIRED_ARRAYS = {
    "chat": {"sample_id", "split", "input_ids", "labels"},\n    "code": {"sample_id", "split", "token_ids"},
    "vision": {"sample_id", "split", "images", "token_ids"},
    "image": {"sample_id", "split", "images", "token_ids"},
    "asr": {"sample_id", "split", "features", "input_lengths", "targets", "target_lengths"},
    "tts": {"sample_id", "split", "token_ids", "frame_to_token", "target_mels", "durations"},
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
    if model_id in {"asr", "tts"} and payload.get("speaker_consent_reviewed") is not True:
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
