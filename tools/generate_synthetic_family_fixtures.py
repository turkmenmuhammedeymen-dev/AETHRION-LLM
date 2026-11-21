"""Create deterministic, explicitly non-production fixtures for all 11 AETHRION models.

These are software-integration fixtures, not capability-training data. Every
record and sidecar is marked synthetic; no human authorship, rights, consent,
privacy review, or training approval is asserted. Never use these fixtures to
open a production training gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np

from aethrion.audio_features import log_mel_spectrogram, read_wav

MODELS = ("chat", "code", "vision", "image", "asr", "tts", "vocoder",
          "3d", "research", "document", "safety")
SPLITS = ("train", "validation", "test")


def split_for(group_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def group_ids_for(seed: int, counts: dict[str, int]) -> dict[str, list[str]]:
    found = {name: [] for name in SPLITS}
    candidate = 0
    while any(len(found[name]) < counts[name] for name in SPLITS):
        group = f"synthetic-group-{seed}-{candidate:08d}"
        name = split_for(group, seed)
        if len(found[name]) < counts[name]:
            found[name].append(group)
        candidate += 1
    return found


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def tagged(row: dict) -> dict:
    return {**row, "synthetic_fixture": True, "production_training_approved": False}


def make_text_rows(model: str, groups: dict[str, list[str]]) -> list[dict]:
    rows = []
    index = 0
    for split in SPLITS:
        for group in groups[split]:
            suffix = f"{split} örneği {index}"
            common = {"id": f"{model}-{index:05d}", "group_id": group}
            if model == "chat":
                row = {**common, "prompt": f"Türkçe bir test sorusunu açıkla: {suffix}.",
                       "response": f"Bu sentetik yazılım testi yanıtıdır; doğrulanmış bilgi iddiası taşımaz. {suffix}."}
            elif model == "code":
                row = {**common, "text": f"# SYNTHETIC TEST ONLY\ndef sample_{index}():\n    return {index}\n"}
            elif model == "research":
                row = {**common, "query": f"Sentetik araştırma sorgusu {suffix}",
                       "passage": f"Bu pasaj yalnızca arama hattı testidir. {suffix}."}
            elif model == "document":
                # One UTF-8 byte keeps the synthetic token/box alignment deterministic.
                row = {**common, "text": "A", "boxes": [[0.0, 0.0, 1.0, 1.0]],
                       "layout_ids": [0], "tag_labels": [0]}
            else:
                # These placeholders exercise tensor shapes only, never SAFE learning.
                decision, reason = (("ALLOW", 0) if index % 3 == 0 else
                                    ("REVIEW", 14) if index % 3 == 1 else ("BLOCK", 1))
                row = {**common, "text": f"Sentetik güvenlik sınıflandırma girdisi {suffix}.",
                       "decision": decision, "reasons": [reason]}
            rows.append(tagged(row))
            index += 1
    return rows


def make_safe_rows(seed: int) -> list[dict]:
    # The preparer requires >=25 samples of each decision in every split.
    # Unique groups prevent leakage; labels are synthetic shape-test placeholders.
    rows = []
    index = 0
    for split in SPLITS:
        for decision, reason in (("ALLOW", 0), ("REVIEW", 14), ("BLOCK", 1)):
            for sample in range(25):
                base = f"safe-{seed}-{split}-{decision}-{sample:02d}"
                group = base
                probe = 0
                while split_for(group, seed) != split:
                    group = f"{base}-{probe}"
                    probe += 1
                rows.append(tagged({
                    "id": f"safety-{index:05d}", "group_id": group,
                    "text": f"Sentetik SAFE test girdisi {index}; karar etiketi test amaçlıdır.",
                    "decision": decision, "reasons": [reason],
                }))
                index += 1
    return rows


def make_audio(root: Path, groups: dict[str, list[str]], sample_rate: int) -> None:
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for split in SPLITS:
        for group in groups[split]:
            # Non-speech tone: never a human voice or speaker recording.
            frequency = 180.0 + (index % 12) * 17.0
            count = sample_rate // 2
            t = np.arange(count, dtype=np.float64) / sample_rate
            samples = (0.12 * np.sin(2.0 * math.pi * frequency * t) * 32767).astype("<i2")
            name = f"tone-{index:04d}.wav"
            with wave.open(str(audio_dir / name), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(samples.tobytes())
            wav_path = audio_dir / name
            # Use the exact same feature extractor and hop lengths as the adapters.
            waveform, rate = read_wav(wav_path, target_sample_rate=sample_rate)
            asr_tts_frames = int(log_mel_spectrogram(
                waveform, sample_rate=rate, n_mels=80, n_fft=640, hop_length=160
            ).shape[0])
            rows.append(tagged({
                "id": f"audio-{index:04d}", "group_id": group,
                "speaker_id": group, "wav_path": f"audio/{name}",
                # A single-token placeholder makes CTC and duration alignment testable.
                "transcript": "a", "durations": [asr_tts_frames],
            }))
            index += 1
    for model in ("asr", "tts", "vocoder"):
        model_rows = rows if model != "vocoder" else [
            {k: v for k, v in row.items() if k not in {"transcript", "durations"}}
            for row in rows
        ]
        write_jsonl(root / f"{model}.jsonl", model_rows)


def make_visual(root: Path, groups: dict[str, list[str]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Install visual extras first: pip install -e '.[visual]'") from exc
    for model, size in (("vision", 224), ("image", 64)):
        folder = root / model / "images"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        index = 0
        for split in SPLITS:
            for group in groups[split]:
                name = f"synthetic-{index:04d}.png"
                image = Image.new("RGB", (size, size), (245, 245, 245))
                draw = ImageDraw.Draw(image)
                margin = 4 + index % max(5, size // 4)
                draw.rectangle((margin, margin, size - margin, size - margin),
                               outline=(20 + index % 180, 70, 130), width=max(1, size // 32))
                draw.text((margin + 2, margin + 2), f"T{index}", fill=(0, 0, 0))
                image.save(folder / name, format="PNG")
                rows.append(tagged({
                    "id": f"{model}-{index:04d}", "group_id": group,
                    "image_path": f"images/{name}",
                    "caption": f"Sentetik geometrik şekil {index}; görsel hat testi.",
                }))
                index += 1
        write_jsonl(root / model / "source.jsonl", rows)

    folder = root / "3d" / "points"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for split in SPLITS:
        for group in groups[split]:
            rng = np.random.default_rng(9000 + index)
            points = rng.uniform(-1.0, 1.0, size=(64, 3)).astype(np.float32)
            name = f"cloud-{index:04d}.npy"
            np.save(folder / name, points, allow_pickle=False)
            rows.append(tagged({
                "id": f"3d-{index:04d}", "group_id": group,
                "point_path": f"points/{name}",
                "text": f"Sentetik normalize nokta bulutu {index}.",
            }))
            index += 1
    write_jsonl(root / "3d" / "source.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic/fixtures")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    if args.sample_rate != 16000:
        parser.error("AETHRION audio fixture adapters currently require 16000 Hz.")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups = group_ids_for(args.seed, {"train": 24, "validation": 3, "test": 3})

    for model in ("chat", "code", "research", "document"):
        write_jsonl(root / f"{model}.jsonl", make_text_rows(model, groups))
    write_jsonl(root / "safety.jsonl", make_safe_rows(args.seed))
    make_audio(root, groups, args.sample_rate)
    make_visual(root, groups)

    manifest = {
        "format": "aethrion-synthetic-fixture-family-v1",
        "seed": args.seed,
        "models": list(MODELS),
        "record_counts": {
            "chat": 30, "code": 30, "vision": 30, "image": 30,
            "asr": 30, "tts": 30, "vocoder": 30, "3d": 30,
            "research": 30, "document": 30, "safety": 225,
        },
        "source_kind": "deterministic_synthetic_software_fixture",
        "not_for_training": True,
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "speaker_consent_reviewed": False,
        "human_reviewed": False,
        "production_training_approved": False,
        "excluded_external_datasets": [
            "Common Voice", "ShapeNet", "CodeSearchNet", "LibriSpeech"
        ],
        "limitations": [
            "Text is template-generated and not a natural-language capability corpus.",
            "Audio consists of non-speech tones, not speech or speaker recordings.",
            "Images are simple geometric test patterns, not semantic vision data.",
            "3D assets are random point clouds, not semantic object scans.",
            "SAFE labels are placeholders, not human-reviewed safety annotations.",
            "Fixtures exist only for parser, tensor-shape, provenance and CI integration tests."
        ],
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(root), "models": len(MODELS),
                      "training_allowed": False, "not_for_training": True}, indent=2))


if __name__ == "__main__":
    main()
