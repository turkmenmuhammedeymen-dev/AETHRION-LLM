"""Build deterministic, disposable synthetic fixtures for all eleven AETHRION models.

These fixtures validate file layouts and preprocessing integrations only. They are
not owner-authored or licensed production data, are never accepted by training
validators, and never change any approval or model-status manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
from pathlib import Path
from typing import Any

MODEL_IDS = (
    "chat", "code", "vision", "image", "asr", "tts", "vocoder",
    "3d", "research", "document", "safety",
)
ORIGIN = "ai_generated_synthetic"

TEXT = [
    "Bir algoritma, bir problemi çözmek için izlenen sonlu adımlardır.",
    "Veri doğrulama, beklenmeyen girdilerin güvenli biçimde ele alınmasına yardım eder.",
    "Bir deneyin sonucu, kullanılan yöntem ve ölçüm koşullarıyla birlikte raporlanmalıdır.",
    "Eğitim ve değerlendirme kümeleri birbirinden bağımsız tutulmalıdır.",
]

def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

def _write_ppm(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = height = 16
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((x * 13 + seed) % 256, (y * 17 + seed * 3) % 256, ((x + y) * 7) % 256))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)

def _write_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, count = 16000, 1600
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = bytearray()
        frequency = 180 + seed * 25
        for index in range(count):
            sample = int(1800 * math.sin(2 * math.pi * frequency * index / rate))
            frames.extend(struct.pack("<h", sample))
        wav.writeframes(bytes(frames))

def _records(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    # CHAT: instruction-response records.
    _jsonl(root / "chat" / "records.jsonl", [
        {"id": f"chat-{i}", "group_id": f"synthetic-chat-{i}", "prompt": f"{TEXT[i]} Bunu kısa biçimde açıkla.", "response": TEXT[(i + 1) % len(TEXT)]}
        for i in range(4)
    ])
    counts["chat"] = 4

    # CODE: small, harmless, original Python snippets.
    _jsonl(root / "code" / "records.jsonl", [
        {"id": f"code-{i}", "group_id": f"synthetic-code-{i}", "text": f"def count_items(items):\n    return len(items)\n\n# Example {i}: {TEXT[i]}"}
        for i in range(4)
    ])
    counts["code"] = 4

    for model in ("vision", "image"):
        rows = []
        for i in range(4):
            asset = f"assets/{model}-{i}.ppm"
            _write_ppm(root / model / asset, i + (0 if model == "vision" else 10))
            rows.append({"id": f"{model}-{i}", "group_id": f"synthetic-{model}-{i}", "image_path": asset, "caption": f"Synthetic geometric color pattern {i}; not a real-world photograph."})
        _jsonl(root / model / "records.jsonl", rows)
        counts[model] = len(rows)

    for model in ("asr", "tts", "vocoder"):
        rows = []
        for i in range(4):
            asset = f"assets/{model}-{i}.wav"
            _write_wav(root / model / asset, i + (1 if model == "asr" else 5 if model == "tts" else 9))
            row: dict[str, Any] = {"id": f"{model}-{i}", "group_id": f"synthetic-{model}-{i}", "speaker_id": f"synthetic-speaker-{i}", "wav_path": asset}
            if model in ("asr", "tts"):
                row["transcript"] = TEXT[i]
            if model == "tts":
                row["durations"] = [max(1, len(TEXT[i]) // 8)]
            rows.append(row)
        _jsonl(root / model / "records.jsonl", rows)
        counts[model] = len(rows)

    rows_3d = []
    for i in range(4):
        asset = f"assets/shape-{i}.json"
        points = [[round(math.cos(j * math.pi / 8) * (1 + i * .1), 5), round(math.sin(j * math.pi / 8), 5), round((j % 3) * .1, 5)] for j in range(16)]
        target = root / "3d" / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"points": points}, separators=(",", ":")) + "\n", encoding="utf-8")
        rows_3d.append({"id": f"3d-{i}", "group_id": f"synthetic-3d-{i}", "point_path": asset, "text": f"Synthetic ring-like point set {i}"})
    _jsonl(root / "3d" / "records.jsonl", rows_3d)
    counts["3d"] = len(rows_3d)

    _jsonl(root / "research" / "records.jsonl", [
        {"id": f"research-{i}", "group_id": f"synthetic-research-{i}", "query": f"Bu ifade hangi kavramı anlatır? {TEXT[i]}", "passage": TEXT[i]}
        for i in range(4)
    ])
    counts["research"] = 4

    _jsonl(root / "document" / "records.jsonl", [
        {"id": f"document-{i}", "group_id": f"synthetic-document-{i}", "text": TEXT[i], "boxes": [[0, 0, 100, 20]], "layout_ids": [1], "tag_labels": [0]}
        for i in range(4)
    ])
    counts["document"] = 4

    _jsonl(root / "safety" / "records.jsonl", [
        {"id": "safety-0", "group_id": "synthetic-safety-0", "text": "Genel bir bilgi talebi için güvenilir kaynakları karşılaştır.", "decision": "ALLOW", "reasons": ["benign_request"]},
        {"id": "safety-1", "group_id": "synthetic-safety-1", "text": "Belirsiz bir istekte önce güvenli bağlamı netleştir.", "decision": "REVIEW", "reasons": ["ambiguous_context"]},
        {"id": "safety-2", "group_id": "synthetic-safety-2", "text": "Gerçek dünyada zarar verebilecek talimatları sağlama.", "decision": "BLOCK", "reasons": ["harmful_operational_request"]},
        {"id": "safety-3", "group_id": "synthetic-safety-3", "text": "Kişisel bilgileri paylaşmadan önce gizlilik riskini değerlendir.", "decision": "ALLOW", "reasons": ["privacy_preserving"]},
    ])
    counts["safety"] = 4
    return counts

def build(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = _records(output)
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "format": "aethrion-all-model-synthetic-fixtures-v1",
        "dataset_id": "aethrion-synthetic-fixtures-all-models-v1",
        "origin": ORIGIN,
        "purpose": "schema_and_pipeline_testing_only",
        "human_review_status": "not_reviewed",
        "rights_holder_attestation": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "speaker_consent_reviewed": False,
        "training_allowed": False,
        "production_training_eligible": False,
        "model_count": len(MODEL_IDS),
        "models": {model: {"record_count": counts[model], "origin": ORIGIN, "training_allowed": False} for model in MODEL_IDS},
        "files": files,
        "warning": "Synthetic fixtures are not production training data. Never copy approval flags into production manifests.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "model_count": len(MODEL_IDS), "record_counts": counts, "file_count": len(files), "training_allowed": False, "origin": ORIGIN}

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate disposable synthetic fixtures for all eleven AETHRION models.")
    parser.add_argument("--output", default="data/synthetic_fixtures")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.output)), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
