"""Fail-closed governance and integrity checks for specialist multimodal datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from aethrion.data_governance import GovernanceError, validate_governance_record

FORMAT = "aethrion-specialist-dataset-v1"
MODEL_MODALITIES = {
    "chat": "text",
    "code": "code",
    "vision": "vision",
    "image": "image",
    "asr": "audio",
    "tts": "audio",
    "vocoder": "audio",
    "3d": "3d",
    "research": "text",
    "document": "document",
    "safety": "text",
}
AUDIO_MODELS = frozenset({"asr", "tts", "vocoder"})
REQUIRED_ARRAYS = {
    "chat": {"sample_id", "group_id", "split", "input_ids", "labels"},
    "code": {"sample_id", "group_id", "split", "token_ids"},
    "vision": {"sample_id", "group_id", "split", "images", "token_ids"},
    "image": {"sample_id", "group_id", "split", "images", "token_ids"},
    "asr": {"sample_id", "group_id", "split", "features", "input_lengths", "targets", "target_lengths"},
    "tts": {"sample_id", "group_id", "split", "token_ids", "token_mask", "token_lengths", "frame_to_token", "frame_mask", "frame_lengths", "target_mels", "durations"},
    "vocoder": {"sample_id", "group_id", "split", "target_mels", "target_waveforms", "frame_lengths", "waveform_mask"},
    "3d": {"sample_id", "group_id", "split", "points", "token_ids"},
    "research": {"sample_id", "group_id", "split", "query_ids", "passage_ids"},
    "document": {"sample_id", "group_id", "split", "token_ids", "boxes", "layout_ids", "tag_labels"},
    "safety": {"sample_id", "group_id", "split", "token_ids", "decision_labels", "reason_labels"},
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
    if candidate.is_absolute() or chr(92) in relative or "://" in relative:
        raise SpecialistDataError("Dataset paths must be relative local paths.")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise SpecialistDataError("Dataset path contains an unsafe component.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SpecialistDataError("Dataset path escapes the declared data root.") from exc
    return resolved


def _validate_modality_shapes(model_id: str, arrays: Any, sample_count: int) -> None:
    """Validate tensor shapes, integer labels and alignment before model creation."""
    if model_id in {"chat", "code"}:
        key = "input_ids" if model_id == "chat" else "token_ids"
        ids = arrays[key]
        if ids.ndim != 2 or ids.shape[1] < 2 or not np.issubdtype(ids.dtype, np.integer):
            raise SpecialistDataError(f"{key} must be an integer [samples, sequence>=2] array.")
        if model_id == "chat":
            labels = arrays["labels"]
            if labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer):
                raise SpecialistDataError("Instruction labels must be an integer array matching input_ids.")
            if np.any((labels < 0) & (labels != -100)):
                raise SpecialistDataError("Instruction labels may only use token IDs or the -100 ignore index.")
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
        if not np.issubdtype(input_lengths.dtype, np.integer) or not np.issubdtype(target_lengths.dtype, np.integer):
            raise SpecialistDataError("ASR input and target lengths must be integer arrays.")
        if np.any(input_lengths < 1) or np.any(input_lengths > features.shape[1]):
            raise SpecialistDataError("ASR input lengths are outside the feature sequence.")
        if np.any(target_lengths < 1) or np.any(target_lengths > targets.shape[1]):
            raise SpecialistDataError("ASR target lengths are outside the padded target sequence.")
    elif model_id == "tts":
        ids, alignment = arrays["token_ids"], arrays["frame_to_token"]
        mels, durations = arrays["target_mels"], arrays["durations"]
        token_mask, token_lengths = arrays["token_mask"], arrays["token_lengths"]
        frame_mask, frame_lengths = arrays["frame_mask"], arrays["frame_lengths"]
        if ids.ndim != 2 or alignment.ndim != 2 or mels.ndim != 3 or durations.shape != ids.shape:
            raise SpecialistDataError("TTS token, alignment, mel and duration arrays have invalid shapes.")
        if alignment.shape != mels.shape[:2] or token_mask.shape != ids.shape or frame_mask.shape != alignment.shape:
            raise SpecialistDataError("TTS masks and alignment must match their padded arrays.")
        if token_lengths.shape != (sample_count,) or frame_lengths.shape != (sample_count,):
            raise SpecialistDataError("TTS token_lengths and frame_lengths must contain one value per sample.")
        if not all(np.issubdtype(value.dtype, np.integer) for value in (alignment, durations, token_lengths, frame_lengths)):
            raise SpecialistDataError("TTS alignment, duration and length arrays must be integers.")
        if np.any(token_lengths < 1) or np.any(token_lengths > ids.shape[1]) or np.any(frame_lengths < 1) or np.any(frame_lengths > alignment.shape[1]):
            raise SpecialistDataError("TTS sequence lengths are outside their padded arrays.")
        expected_token_mask = np.arange(ids.shape[1])[None, :] < token_lengths[:, None]
        expected_frame_mask = np.arange(alignment.shape[1])[None, :] < frame_lengths[:, None]
        if np.any((token_mask != 0) & (token_mask != 1)) or np.any((frame_mask != 0) & (frame_mask != 1)):
            raise SpecialistDataError("TTS token and frame masks must be binary.")
        if not np.array_equal(token_mask.astype(bool), expected_token_mask) or not np.array_equal(frame_mask.astype(bool), expected_frame_mask):
            raise SpecialistDataError("TTS masks do not match the declared sequence lengths.")
        if np.any(durations[token_mask.astype(bool)] < 1) or np.any(durations[~token_mask.astype(bool)] != 0):
            raise SpecialistDataError("TTS durations must be positive for real tokens and zero for padding.")
        for index in range(sample_count):
            token_count, frame_count = int(token_lengths[index]), int(frame_lengths[index])
            aligned = alignment[index, :frame_count]
            if np.any(aligned < 0) or np.any(aligned >= token_count):
                raise SpecialistDataError("TTS alignment references a padded or missing token.")
            if int(durations[index, :token_count].sum()) != frame_count:
                raise SpecialistDataError("TTS valid-token durations must sum to frame_lengths.")
            observed = np.bincount(aligned.astype(np.int64), minlength=token_count)
            if not np.array_equal(observed, durations[index, :token_count]):
                raise SpecialistDataError("TTS frame-to-token alignment and durations disagree.")
    elif model_id == "vocoder":
        mels, audio = arrays["target_mels"], arrays["target_waveforms"]
        frame_lengths, waveform_mask = arrays["frame_lengths"], arrays["waveform_mask"]
        if mels.ndim != 3 or audio.ndim != 2 or frame_lengths.shape != (sample_count,) or waveform_mask.shape != audio.shape:
            raise SpecialistDataError("Vocoder requires aligned mel, waveform, frame-length and waveform-mask arrays.")
        if not np.issubdtype(frame_lengths.dtype, np.integer):
            raise SpecialistDataError("Vocoder frame_lengths must be integers.")
        if np.any(frame_lengths < 1) or np.any(frame_lengths > mels.shape[1]):
            raise SpecialistDataError("Vocoder frame_lengths are outside the mel array.")
        if np.any((waveform_mask != 0) & (waveform_mask != 1)):
            raise SpecialistDataError("Vocoder waveform_mask must be binary.")
    elif model_id == "3d":
        points, ids = arrays["points"], arrays["token_ids"]
        if points.ndim != 3 or points.shape[2] != 3 or ids.ndim != 2:
            raise SpecialistDataError("3D data requires points [N,points,3] and token_ids [N,L].")
        if points.shape[1] < 16 or not np.issubdtype(ids.dtype, np.integer):
            raise SpecialistDataError("3D point clouds or conditioning token IDs are invalid.")
    elif model_id == "research":
        query_ids, passage_ids = arrays["query_ids"], arrays["passage_ids"]
        if query_ids.ndim != 2 or passage_ids.ndim != 2:
            raise SpecialistDataError("Research query_ids and passage_ids must be padded 2D arrays.")
        if not np.issubdtype(query_ids.dtype, np.integer) or not np.issubdtype(passage_ids.dtype, np.integer):
            raise SpecialistDataError("Research token arrays must use integer dtypes.")
    elif model_id == "document":
        ids, boxes, layouts, tags = arrays["token_ids"], arrays["boxes"], arrays["layout_ids"], arrays["tag_labels"]
        if ids.ndim != 2 or boxes.shape != (*ids.shape, 4) or layouts.shape != ids.shape or tags.shape != ids.shape:
            raise SpecialistDataError("Document token, layout, boxes and tag arrays have inconsistent shapes.")
        if not np.issubdtype(layouts.dtype, np.integer) or not np.issubdtype(tags.dtype, np.integer):
            raise SpecialistDataError("Document layout and tag labels must be integer arrays.")
        if np.any(boxes < 0) or np.any(boxes > 1):
            raise SpecialistDataError("Document boxes must be normalized to [0,1].")
    elif model_id == "safety":
        ids, decisions, reasons = arrays["token_ids"], arrays["decision_labels"], arrays["reason_labels"]
        if ids.ndim != 2 or decisions.shape != (sample_count,) or reasons.ndim != 2 or reasons.shape[0] != sample_count:
            raise SpecialistDataError("Safety token, decision and reason arrays have inconsistent shapes.")
        if not np.issubdtype(decisions.dtype, np.integer):
            raise SpecialistDataError("Safety decision labels must be integer values.")
        if not np.issubdtype(reasons.dtype, np.integer):
            raise SpecialistDataError("Safety reason labels must be integer multi-hot values.")
        if np.any(decisions < 0) or np.any(decisions > 2):
            raise SpecialistDataError("Safety decision labels must be ALLOW=0, REVIEW=1 or BLOCK=2.")
        if np.any(reasons < 0) or np.any(reasons > 1):
            raise SpecialistDataError("Safety reason labels must be binary multi-hot values.")


def validate_specialist_dataset(
    manifest_path: str | Path,
    *,
    require_training_approval: bool = True,
) -> dict[str, Any]:
    """Validate source licenses, local hashes, NPZ safety, modality shapes and split integrity."""
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
    if type(payload.get("training_allowed")) is not bool or type(payload.get("owner_approval")) is not bool:
        raise SpecialistDataError("training_allowed and owner_approval must be exact JSON booleans.")
    if require_training_approval and (
        payload["training_allowed"] is not True or payload["owner_approval"] is not True
    ):
        raise SpecialistDataError("Training blocked: specialist dataset and owner approvals are not true.")
    if require_training_approval:
        try:
            validate_governance_record(payload.get("governance"), model_id=model_id)
        except GovernanceError as exc:
            raise SpecialistDataError(str(exc)) from exc
    if payload.get("privacy_reviewed") is not True:
        raise SpecialistDataError("Training blocked: privacy review is not approved.")
    if model_id in AUDIO_MODELS and payload.get("speaker_consent_reviewed") is not True:
        raise SpecialistDataError("Training blocked: speaker consent review is not approved.")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SpecialistDataError("At least one documented training source is required.")
    seen_source_ids: set[str] = set()
    seen_source_paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise SpecialistDataError("Each source review must be an object.")
        required_text = ("source_id", "source_uri", "source_path", "license", "permitted_use", "evidence_uri")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in required_text):
            raise SpecialistDataError("Each source requires an ID, URI, path, license, permitted use and evidence.")
        source_id = source["source_id"].strip()
        source_path_value = source["source_path"].strip()
        if source_id in seen_source_ids:
            raise SpecialistDataError(f"Duplicate specialist source_id: {source_id}")
        if source_path_value in seen_source_paths:
            raise SpecialistDataError(f"Duplicate specialist source_path: {source_path_value}")
        seen_source_ids.add(source_id)
        seen_source_paths.add(source_path_value)
        digest = source.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest.lower()):
            raise SpecialistDataError("Each source requires a valid SHA-256.")
        source_file = _safe_local_path(manifest_file.parent.resolve(), source["source_path"])
        if not source_file.is_file() or source_file.stat().st_size == 0:
            raise SpecialistDataError("A reviewed source file is missing or empty.")
        if sha256_file(source_file) != digest.lower():
            raise SpecialistDataError("A reviewed source file SHA-256 mismatch.")
        if source.get("allow_training") is not True:
            raise SpecialistDataError("Training blocked: one or more source licenses are not approved.")
        if source.get("privacy_reviewed") is not True:
            raise SpecialistDataError("Training blocked: one or more sources lack privacy approval.")
        if model_id in AUDIO_MODELS and source.get("speaker_consent_verified") is not True:
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
            # Materialize every NPZ array before the archive context closes.
            # NpzFile lazily reads from its ZIP handle and cannot be used afterwards.
            arrays = {key: np.array(arrays[key], copy=True) for key in keys}
            sample_ids = arrays["sample_id"]
            group_ids = arrays["group_id"]
            splits = arrays["split"]
            if (
                sample_ids.ndim != 1 or group_ids.ndim != 1 or splits.ndim != 1
                or len(sample_ids) == 0
                or len(sample_ids) != len(group_ids) or len(sample_ids) != len(splits)
            ):
                raise SpecialistDataError("sample_id, group_id and split arrays must be non-empty 1D arrays of equal length.")
            ids = [str(item) for item in sample_ids.tolist()]
            group_names = [str(item) for item in group_ids.tolist()]
            split_names = [str(item) for item in splits.tolist()]
            if len(set(ids)) != len(ids) or any(not item for item in ids):
                raise SpecialistDataError("sample_id values must be unique and non-empty.")
            if any(not SHA256.fullmatch(item) for item in group_names):
                raise SpecialistDataError(
                    "Every group_id must be a pseudonymous lowercase SHA-256 value."
                )
            group_split = {}
            for group_name, split_name in zip(group_names, split_names):
                previous = group_split.setdefault(group_name, split_name)
                if previous != split_name:
                    raise SpecialistDataError(
                        f"Provenance group {group_name!r} crosses {previous} and {split_name} splits."
                    )
            if set(split_names) - {"train", "validation", "test"}:
                raise SpecialistDataError("split values must be train, validation or test.")
            counts = {name: split_names.count(name) for name in ("train", "validation", "test")}
            if any(counts[name] == 0 for name in counts):
                raise SpecialistDataError("Dataset requires non-empty train, validation and test splits.")
            sample_count = len(ids)
            _validate_modality_shapes(model_id, arrays, sample_count)
            if model_id == "safety":
                minimum = payload.get("safety_min_per_class")
                if type(minimum) is not int or minimum < 25:
                    raise SpecialistDataError(
                        "Safety datasets must declare safety_min_per_class >= 25 for every decision in every split."
                    )
                decisions = np.asarray(arrays["decision_labels"])
                split_array = np.asarray(split_names)
                for split_name in ("train", "validation", "test"):
                    split_decisions = decisions[split_array == split_name]
                    class_counts = np.bincount(split_decisions.astype(np.int64), minlength=3)
                    if np.any(class_counts < minimum):
                        raise SpecialistDataError(
                            f"Safety {split_name} split must contain at least {minimum} samples for ALLOW, REVIEW and BLOCK."
                        )
            for key in keys - {"sample_id", "group_id", "split"}:
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

    tokenizer_path = None
    tokenizer_hash = None
    tokenizer_vocab_size = None
    if model_id != "vocoder":
        raw_tokenizer_path = payload.get("tokenizer_path")
        tokenizer_hash = payload.get("tokenizer_sha256")
        if not isinstance(raw_tokenizer_path, str) or not raw_tokenizer_path.strip():
            raise SpecialistDataError("Tokenizer path is required for this specialist model.")
        if not isinstance(tokenizer_hash, str) or not SHA256.fullmatch(tokenizer_hash.lower()):
            raise SpecialistDataError("Tokenizer SHA-256 is required for this specialist model.")
        tokenizer_path = _safe_local_path(root, raw_tokenizer_path)
        if not tokenizer_path.is_file() or tokenizer_path.stat().st_size == 0:
            raise SpecialistDataError("Specialist tokenizer file is missing or empty.")
        if sha256_file(tokenizer_path) != tokenizer_hash.lower():
            raise SpecialistDataError("Specialist tokenizer SHA-256 mismatch.")
        try:
            from aethrion.tokenizer import ByteBPETokenizer
            tokenizer_vocab_size = ByteBPETokenizer.load(tokenizer_path).vocab_size
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            raise SpecialistDataError("Specialist tokenizer is invalid or unreadable.") from exc
        token_arrays = {"input_ids", "token_ids", "query_ids", "passage_ids", "targets"} & set(keys)
        for key in token_arrays:
            values = np.asarray(arrays[key])
            if not np.issubdtype(values.dtype, np.integer):
                raise SpecialistDataError(f"{key} must use an integer tokenizer-ID dtype.")
            if np.any(values < 0) or np.any(values >= tokenizer_vocab_size):
                raise SpecialistDataError(
                    f"{key} contains token IDs outside the verified tokenizer vocabulary."
                )
        if model_id == "chat":
            labels = np.asarray(arrays["labels"])
            response_tokens = labels[labels != -100]
            if not response_tokens.size or np.any(response_tokens < 0) or np.any(response_tokens >= tokenizer_vocab_size):
                raise SpecialistDataError(
                    "Instruction response labels contain IDs outside the verified tokenizer vocabulary."
                )

    return {
        "model_id": model_id,
        "modality": payload["modality"],
        "dataset_path": str(file_path),
        "dataset_sha256": actual_hash,
        "sample_count": sample_count,
        "split_counts": counts,
        "source_count": len(sources),
        "group_count": len(set(group_names)),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_vocab_size": tokenizer_vocab_size,
    }
