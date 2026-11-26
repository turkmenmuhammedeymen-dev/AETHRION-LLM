"""Full (not sampled) structural audit of every binary asset in the prepared specialist datasets.

Read-only. Checks existence, decodability, format parameters, NaN/Inf, duplicates, caption/transcript
presence and the manifest hashes. It does not judge semantic quality and never approves anything.
"""
from __future__ import annotations

import hashlib
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/specialists/manifests"
AUTHORED = {"image": "image", "vision": "vision", "asr": "audio", "tts": "audio", "vocoder": "audio", "3d": "3d"}


def asset(model, rel):
    return REPO / "data/authored" / AUTHORED[model] / rel


def rows(model):
    return [json.loads(l) for l in (ROOT / model / "source.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def audit_images(model, issues):
    from PIL import Image
    sizes, hashes = Counter(), Counter()
    n = 0
    for r in rows(model):
        p = asset(model, r["image_path"])
        n += 1
        if not p.is_file():
            issues.append(f"{model}:{r['id']} missing {r['image_path']}"); continue
        try:
            with Image.open(p) as im:
                im.load(); sizes[(im.size, im.mode)] += 1
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{model}:{r['id']} undecodable: {exc}")
        hashes[hashlib.sha256(p.read_bytes()).hexdigest()] += 1
        if not r.get("caption", "").strip():
            issues.append(f"{model}:{r['id']} empty caption")
    dup = sum(c - 1 for c in hashes.values() if c > 1)
    return {"files": n, "formats": {f"{k[0]} {k[1]}": v for k, v in sizes.items()}, "byte_duplicates": dup}


def audit_audio(model, issues):
    rates, chans, widths, durs = Counter(), Counter(), Counter(), []
    silent = n = 0
    hashes = Counter()
    for r in rows(model):
        p = asset(model, r["wav_path"])
        n += 1
        if not p.is_file():
            issues.append(f"{model}:{r['id']} missing {r['wav_path']}"); continue
        try:
            with wave.open(str(p)) as w:
                rates[w.getframerate()] += 1; chans[w.getnchannels()] += 1; widths[w.getsampwidth()] += 1
                frames = w.readframes(w.getnframes()); durs.append(w.getnframes() / w.getframerate())
            x = np.frombuffer(frames, dtype=np.int16) if widths and w.getsampwidth() == 2 else np.zeros(1)
            if x.size == 0 or np.abs(x).max() < 50:
                silent += 1; issues.append(f"{model}:{r['id']} silent/empty")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{model}:{r['id']} bad wav: {exc}")
        hashes[hashlib.sha256(p.read_bytes()).hexdigest()] += 1
        if "transcript" in r and not r["transcript"].strip():
            issues.append(f"{model}:{r['id']} empty transcript")
    return {"files": n, "sample_rates": dict(rates), "channels": dict(chans), "sample_widths": dict(widths),
            "duration_s": [round(min(durs), 3), round(max(durs), 3)] if durs else None, "silent": silent,
            "byte_duplicates": sum(c - 1 for c in hashes.values() if c > 1)}


def audit_3d(issues):
    shapes, n = Counter(), 0
    for r in rows("3d"):
        p = asset("3d", r["point_path"])
        n += 1
        if not p.is_file():
            issues.append(f"3d:{r['id']} missing {r['point_path']}"); continue
        a = np.load(p)
        shapes[(a.shape, str(a.dtype))] += 1
        if not np.isfinite(a).all():
            issues.append(f"3d:{r['id']} NaN/Inf")
        elif np.abs(a).max() > 10:
            issues.append(f"3d:{r['id']} coordinate range {float(np.abs(a).max()):.2f}")
        if not r.get("text", "").strip():
            issues.append(f"3d:{r['id']} empty text")
    return {"files": n, "shapes": {f"{k[0]} {k[1]}": v for k, v in shapes.items()}}


def audit_document(issues):
    n = 0
    for r in rows("document"):
        n += 1
        b = np.asarray(r.get("boxes", []), dtype=float)
        if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all() or (b < 0).any() or (b > 1).any():
            issues.append(f"document:{r['id']} invalid boxes")
        if not r.get("text", "").strip():
            issues.append(f"document:{r['id']} empty text")
    return {"records": n}


def audit_npz(model, issues):
    m = json.loads((ROOT / f"{model}.json").read_text(encoding="utf-8"))
    p = ROOT / model / "dataset.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != m["dataset_sha256"]:
        issues.append(f"{model}: dataset.npz sha256 mismatch with manifest")
    d = np.load(p)
    for k in d.files:
        if d[k].dtype.kind == "f" and not np.isfinite(d[k]).all():
            issues.append(f"{model}: dataset.npz array {k} has NaN/Inf")
    return {"arrays": {k: list(d[k].shape) for k in d.files}}


def main() -> int:
    report, issues = {}, []
    report["image"] = audit_images("image", issues); report["vision"] = audit_images("vision", issues)
    for m in ("asr", "tts", "vocoder"):
        report[m] = audit_audio(m, issues)
    report["3d"] = audit_3d(issues); report["document"] = audit_document(issues)
    for m in ("image", "vision", "asr", "tts", "vocoder", "3d", "document"):
        report[m]["npz"] = audit_npz(m, issues)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"ISSUES: {len(issues)}")
    for i in issues[:60]:
        print(" -", i)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
