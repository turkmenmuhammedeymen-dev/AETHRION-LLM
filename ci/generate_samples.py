"""Generate review samples from the verified checkpoints for HUMAN listening/looking (MOS, image safety).

Read-only: pulls checkpoints, never trains or changes release status. Output files are blind-named
(sample_001.wav ...) with the answer key in key.csv, and a ratings.csv template for the reviewer.
All data is synthetic; these samples measure the synthetic-data models only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAN = ROOT / "data" / "specialists" / "manifests"
SR = 16000


def _load(model, scratch, api):
    from aethrion import artifact_sync
    from aethrion.specialist_train import _build_model, _load_checkpoint
    cfg = json.loads((ROOT / "configs" / "specialists" / f"{model}.json").read_text(encoding="utf-8"))
    out = Path(scratch) / model
    artifact_sync.pull_model_artifacts(model, out, api=api)
    state = _load_checkpoint(out / "checkpoint_last.pt", torch.device("cpu"))
    net = _build_model(model, cfg["model_config"])
    net.load_state_dict(state["model_state"])
    return net.eval()


STATS = []


def _wav(path, x):
    x = np.clip(np.asarray(x, dtype=np.float32), -1, 1)
    STATS.append(f"{Path(path).name} seconds={len(x)/SR:.2f} rms={float(np.sqrt((x**2).mean())):.5f} peak={float(abs(x).max()):.5f}")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    owner = api.whoami().get("name")
    os.environ.setdefault("AETHRION_ARTIFACT_REPO", f"{owner}/aethrion-artifacts")
    out, scratch = Path(a.out), Path("/tmp/aethrion-samples")
    for d in ("audio", "images", "pointclouds"):
        (out / d).mkdir(parents=True, exist_ok=True)
    key, notes = [], []

    # --- TTS -> vocoder, plus vocoder copy-synthesis from real target mels
    try:
        tts, voc = _load("tts", scratch, api), _load("vocoder", scratch, api)
        d = np.load(MAN / "tts" / "dataset.npz", allow_pickle=False)
        rows = [json.loads(l) for l in (MAN / "tts" / "source.jsonl").read_text(encoding="utf-8").splitlines()]
        test = np.flatnonzero(d["split"].astype(str) == "test")[: a.n]
        with torch.no_grad():
            for i in test:
                n = int(d["token_lengths"][i])
                ids = torch.from_numpy(d["token_ids"][i:i + 1, :n]).long()
                dur = tts.predict_durations(ids)
                align, _ = tts.durations_to_alignment(dur)
                mel = tts(ids, align).transpose(1, 2)
                STATS.append(f"tts mel mean={float(mel.mean()):.3f} std={float(mel.std()):.3f} real_mean={float(d['target_mels'][i].mean()):.3f} real_std={float(d['target_mels'][i].std()):.3f}")
                _wav(out / "audio" / f"sample_{len(key) + 1:03d}.wav", voc(mel)[0].numpy())
                key.append((f"sample_{len(key) + 1:03d}.wav", "tts+vocoder", rows[int(i)]["transcript"]))
        with torch.no_grad():
            vd = np.load(MAN / "vocoder" / "dataset.npz", allow_pickle=False)
            for i in np.flatnonzero(vd["split"].astype(str) == "test")[: a.n]:
                m = torch.from_numpy(vd["target_mels"][i:i + 1, : int(vd["frame_lengths"][i])]).transpose(1, 2)
                _wav(out / "audio" / f"sample_{len(key) + 1:03d}.wav", voc(m)[0].numpy())
                key.append((f"sample_{len(key) + 1:03d}.wav", "vocoder copy-synthesis", ""))
    except Exception as exc:  # report, never hide
        notes.append(f"audio failed: {type(exc).__name__}: {exc}")

    # --- IMAGE and 3D from test-split prompts
    try:
        from PIL import Image
        img = _load("image", scratch, api)
        d = np.load(MAN / "image" / "dataset.npz", allow_pickle=False)
        rows = [json.loads(l) for l in (MAN / "image" / "source.jsonl").read_text(encoding="utf-8").splitlines()]
        for i in np.flatnonzero(d["split"].astype(str) == "test")[: a.n]:
            ids = torch.from_numpy(d["token_ids"][i:i + 1]).long()
            with torch.no_grad():
                x = img.generate(ids, inference_steps=50)[0].clamp(-1, 1)
            arr = ((x.permute(1, 2, 0).numpy() + 1) * 127.5).round().astype("uint8")
            name = f"image_{len(key) + 1:03d}.png"
            Image.fromarray(arr).save(out / "images" / name)
            key.append((name, "image", rows[int(i)].get("caption", "")))
    except Exception as exc:
        notes.append(f"image failed: {type(exc).__name__}: {exc}")
    try:
        m3 = _load("3d", scratch, api)
        d = np.load(MAN / "3d" / "dataset.npz", allow_pickle=False)
        rows = [json.loads(l) for l in (MAN / "3d" / "source.jsonl").read_text(encoding="utf-8").splitlines()]
        for i in np.flatnonzero(d["split"].astype(str) == "test")[: a.n]:
            ids = torch.from_numpy(d["token_ids"][i:i + 1]).long()
            with torch.no_grad():
                pts = m3.generate(ids, inference_steps=50)[0].numpy()
            name = f"cloud_{len(key) + 1:03d}.ply"
            with open(out / "pointclouds" / name, "w", encoding="ascii") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                f.writelines(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n" for p in pts)
            key.append((name, "3d", rows[int(i)].get("caption", rows[int(i)].get("text", ""))))
    except Exception as exc:
        notes.append(f"3d failed: {type(exc).__name__}: {exc}")

    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("file", "kind", "reference_text")] + key)
    with open(out / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "mos_1_to_5", "intelligible_yes_no", "unsafe_or_inappropriate_yes_no", "notes"])
        w.writerows([(k[0], "", "", "", "") for k in key])
    (out / "README.txt").write_text(
        "Rate each file in ratings.csv: MOS 1 (bad) to 5 (excellent), intelligible (audio), unsafe (images).\n"
        "Listen/look before opening key.csv. Synthetic-data models only.\n" + "\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(STATS))
    print(f"samples={len(key)} notes={notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
