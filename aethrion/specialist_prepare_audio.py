"""Prepare consent-gated ASR, TTS and Vocoder datasets from local PCM WAV files.

The adapter creates model-shaped NPZ arrays and provenance hashes, but never
approves licenses, privacy or speaker consent. All approval flags remain false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import torch

from aethrion.audio_features import log_mel_spectrogram, read_wav
from aethrion.specialist_prepare import _group_hash, _group_id, _split
from aethrion.specialist_data import AUDIO_MODELS
from aethrion.data_governance import review_template
from aethrion.tokenizer import ByteBPETokenizer


class AudioPreparationError(ValueError):
    """Invalid audio record, unsupported WAV or unsafe source path."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise AudioPreparationError("Audio JSONL is missing or empty.")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AudioPreparationError(f"Invalid JSON on line {line_no}.") from exc
        if not isinstance(item, dict):
            raise AudioPreparationError(f"Line {line_no} must be a JSON object.")
        rows.append(item)
    ids = [str(row.get("id", "")).strip() for row in rows]
    if len(rows) < 3 or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise AudioPreparationError("At least three records with unique, non-empty IDs are required.")
    return rows


def _local_audio(root: Path, value: Any, row_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AudioPreparationError(f"Record {row_id!r} requires wav_path.")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or "\\" in value
        or "://" in value
        or re.match(r"^[A-Za-z]:", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise AudioPreparationError("Audio paths must be safe relative local paths.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AudioPreparationError(f"Audio path for {row_id!r} escapes the source folder.") from exc
    if not resolved.is_file():
        raise AudioPreparationError(f"Audio file for {row_id!r} does not exist.")
    return resolved


def _safe_id(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:48] or "sample"
    return f"{stem}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def prepare_audio_dataset(
    input_path: str | Path,
    *,
    model_id: str,
    output_root: str | Path,
    license_name: str,
    permitted_use: str,
    evidence_uri: str,
    source_uri: str,
    seed: int = 1337,
    vocab_size: int = 2048,
    n_mels: int = 80,
    sample_rate: int = 16000,
    hop_length: int | None = None,
) -> dict[str, Any]:
    if model_id not in AUDIO_MODELS:
        raise AudioPreparationError("model_id must be asr, tts or vocoder.")
    if hop_length is None:
        hop_length = 256 if model_id == "vocoder" else 160
    if not all(isinstance(value, str) and value.strip()
               for value in (license_name, permitted_use, evidence_uri, source_uri)):
        raise AudioPreparationError("License, permitted use, evidence and canonical source URI are required.")
    if vocab_size < 256 or n_mels < 16 or sample_rate != 16000 or hop_length < 80:
        raise AudioPreparationError("Use byte-BPE >=256, n_mels >=16, 16 kHz audio and hop_length >=80.")
    source = Path(input_path).resolve()
    rows = _read_rows(source)
    root = source.parent.resolve()
    transcripts = []
    decoded = []
    copied_sources = []
    group_ids = [_group_id(row, fallback="speaker_id") for row in rows]
    split_names = [_split(group_id, seed) for group_id in group_ids]
    for row in rows:
        row_id = str(row["id"])
        wav_path = _local_audio(root, row.get("wav_path"), row_id)
        waveform, rate = read_wav(wav_path, target_sample_rate=sample_rate)
        mel = log_mel_spectrogram(
            waveform, sample_rate=rate, n_mels=n_mels,
            n_fft=max(400, hop_length * 4), hop_length=hop_length,
        ).cpu().numpy().astype(np.float32)
        if mel.shape[0] < 4 or not np.isfinite(mel).all():
            raise AudioPreparationError(f"Audio for {row_id!r} produced invalid or too-short mel features.")
        transcript = row.get("transcript")
        token_ids = None
        if model_id in {"asr", "tts"}:
            if not isinstance(transcript, str) or not transcript.strip():
                raise AudioPreparationError(f"Record {row_id!r} requires a non-empty transcript.")
            if split_names[len(decoded)] == "train":
                transcripts.append(transcript.strip())
        decoded.append((row, wav_path, waveform, mel))

    tokenizer = None
    if model_id in {"asr", "tts"}:
        if not transcripts:
            raise AudioPreparationError("No train-split transcripts are available to train the tokenizer.")
        tokenizer = ByteBPETokenizer()
        tokenizer.train("\n".join(transcripts), vocab_size=vocab_size)

    token_sequences = []
    if tokenizer is not None:
        for row, _, _, _ in decoded:
            token_ids = tokenizer.encode(str(row["transcript"]).strip())
            if not token_ids or 0 in token_ids:
                raise AudioPreparationError("Transcript produced empty tokens or CTC blank token ID 0.")
            token_sequences.append(token_ids)

    destination = Path(output_root).resolve() / "manifests" / model_id
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_copy = destination / "source.jsonl"
    shutil.copyfile(source, source_copy)
    manifest_sources = [{
        "source_id": f"{model_id}-metadata-jsonl",
        "source_uri": source_uri,
        "source_path": f"{model_id}/source.jsonl",
        "sha256": _sha256(source_copy),
        "license": license_name,
        "permitted_use": permitted_use,
        "evidence_uri": evidence_uri,
        "allow_training": False,
        "privacy_reviewed": False,
        "speaker_consent_verified": False,
    }]

    features = [item[3] for item in decoded]
    frame_lengths = np.asarray([item.shape[0] for item in features], dtype=np.int64)
    max_frames = int(frame_lengths.max())
    padded_mels = np.zeros((len(rows), max_frames, n_mels), dtype=np.float32)
    frame_mask = np.zeros((len(rows), max_frames), dtype=np.bool_)
    for index, feature in enumerate(features):
        padded_mels[index, :feature.shape[0]] = feature
        frame_mask[index, :feature.shape[0]] = True

    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray([str(row["id"]) for row in rows], dtype=np.str_),
        "group_id": np.asarray([_group_hash(value) for value in group_ids], dtype=np.str_),
        "split": np.asarray(split_names, dtype=np.str_),
    }
    if any(name not in split_names for name in ("train", "validation", "test")):
        raise AudioPreparationError("Split has an empty partition; add samples or choose a different seed.")
    arrays["target_mels"] = padded_mels
    arrays["frame_lengths"] = frame_lengths

    if model_id == "asr":
        if n_mels != 80:
            raise AudioPreparationError("Current ASR config requires exactly 80 mel bins.")
        max_tokens = max(map(len, token_sequences))
        targets = np.zeros((len(rows), max_tokens), dtype=np.int64)
        target_lengths = np.asarray([len(item) for item in token_sequences], dtype=np.int64)
        input_lengths = frame_lengths.copy()
        for index, tokens in enumerate(token_sequences):
            encoded_frames = (int(input_lengths[index]) + 3) // 4
            if len(tokens) > encoded_frames:
                raise AudioPreparationError(f"ASR transcript for {rows[index]['id']!r} is too long for its audio.")
            targets[index, :len(tokens)] = tokens
        arrays.update({
            "features": padded_mels,
            "input_lengths": input_lengths,
            "targets": targets,
            "target_lengths": target_lengths,
        })
        arrays.pop("target_mels")
        arrays.pop("frame_lengths")
    elif model_id == "tts":
        if n_mels != 80:
            raise AudioPreparationError("Current TTS config requires exactly 80 mel bins.")
        durations_rows = []
        for row, tokens, feature in zip(rows, token_sequences, features):
            durations = row.get("durations")
            if not isinstance(durations, list) or len(durations) != len(tokens) or any(type(x) is not int or x < 1 for x in durations):
                raise AudioPreparationError(
                    f"TTS record {row['id']!r} needs one positive frame duration per byte-BPE token."
                )
            if sum(durations) != feature.shape[0]:
                raise AudioPreparationError(f"TTS durations for {row['id']!r} must sum to its mel frame count.")
            durations_rows.append(durations)
        max_tokens = max(map(len, token_sequences))
        token_ids = np.zeros((len(rows), max_tokens), dtype=np.int64)
        token_mask = np.zeros((len(rows), max_tokens), dtype=np.bool_)
        durations = np.zeros((len(rows), max_tokens), dtype=np.int64)
        alignment = np.zeros((len(rows), max_frames), dtype=np.int64)
        for index, (tokens, duration) in enumerate(zip(token_sequences, durations_rows)):
            token_ids[index, :len(tokens)] = tokens
            token_mask[index, :len(tokens)] = True
            durations[index, :len(tokens)] = duration
            aligned = np.repeat(np.arange(len(tokens), dtype=np.int64), duration)
            alignment[index, :len(aligned)] = aligned
        arrays.update({
            "token_ids": token_ids,
            "token_mask": token_mask,
            "token_lengths": np.asarray([len(item) for item in token_sequences], dtype=np.int64),
            "frame_to_token": alignment,
            "frame_mask": frame_mask,
            "durations": durations,
        })
    else:
        if n_mels != 80:
            raise AudioPreparationError("Current Vocoder config requires exactly 80 mel bins.")
        hop_samples = max_frames * hop_length
        waveforms = np.zeros((len(rows), hop_samples), dtype=np.float32)
        waveform_mask = np.zeros((len(rows), hop_samples), dtype=np.bool_)
        for index, (_, _, waveform, _) in enumerate(decoded):
            samples = waveform.cpu().numpy().astype(np.float32)
            usable = min(len(samples), hop_samples)
            waveforms[index, :usable] = samples[:usable]
            waveform_mask[index, :usable] = True
        arrays["target_waveforms"] = waveforms
        arrays["waveform_mask"] = waveform_mask

    for row, wav_path, _, _ in decoded:
        safe_name = _safe_id(str(row["id"])) + ".wav"
        copied = audio_dir / safe_name
        shutil.copyfile(wav_path, copied)
        copied_sources.append(copied)
        manifest_sources.append({
            "source_id": f"audio-{_safe_id(str(row['id']))}",
            "source_uri": str(row.get("source_uri") or source_uri),
            "source_path": f"{model_id}/audio/{safe_name}",
            "sha256": _sha256(copied),
            "license": license_name,
            "permitted_use": permitted_use,
            "evidence_uri": evidence_uri,
            "allow_training": False,
            "privacy_reviewed": False,
            "speaker_consent_verified": False,
        })

    dataset = destination / "dataset.npz"
    np.savez_compressed(dataset, **arrays)
    tokenizer_path = destination / "tokenizer.json"
    if tokenizer is not None:
        tokenizer.save(tokenizer_path)
    manifest_path = destination.parent / f"{model_id}.json"
    manifest = {
        "format": "aethrion-specialist-dataset-v1",
        "dataset_id": f"{model_id}-prepared-{_sha256(dataset)[:12]}",
        "model_id": model_id,
        "modality": "audio",
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "speaker_consent_reviewed": False,
        "governance": review_template(model_id),
        "dataset_path": f"{model_id}/dataset.npz",
        "dataset_sha256": _sha256(dataset),
        "sources": manifest_sources,
        "preparation": {
            "tool": "aethrion.specialist_prepare_audio",
            "seed": seed,
            "sample_rate": sample_rate,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "record_count": len(rows),
            "split_counts": {name: split_names.count(name) for name in ("train", "validation", "test")},
            "automatic_approval": False,
        },
    }
    if tokenizer is not None:
        manifest["tokenizer_path"] = f"{model_id}/tokenizer.json"
        manifest["tokenizer_sha256"] = _sha256(tokenizer_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "model_id": model_id,
        "records": len(rows),
        "split_counts": manifest["preparation"]["split_counts"],
        "dataset": str(dataset),
        "manifest": str(manifest_path),
        "training_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare consent-gated AETHRION speech model datasets.")
    parser.add_argument("--input", required=True, help="JSONL with id, wav_path, and transcript/durations as required.")
    parser.add_argument("--model", required=True, choices=("asr", "tts", "vocoder"))
    parser.add_argument("--output-root", default="data/specialists")
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--permitted-use", required=True)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=None)
    args = parser.parse_args()
    result = prepare_audio_dataset(
        args.input, model_id=args.model, output_root=args.output_root,
        license_name=args.license_name, permitted_use=args.permitted_use,
        evidence_uri=args.evidence_uri, source_uri=args.source_uri, seed=args.seed,
        vocab_size=args.vocab_size, n_mels=args.n_mels, sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
