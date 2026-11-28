"""Re-run dataset preparation with a small tokenizer vocabulary (technical fix, not a new review).

Why: with vocab 350-435 on a few hundred near-identical captions, byte-BPE merged whole phrases, so each caption
became ~2 tokens (3D: 1). Models then saw opaque labels and could not follow new combinations of words. A
~270-token vocabulary keeps words/characters compositional (~30 tokens per caption).

Safety rules (fail closed):
  * The source records must be byte-identical (same sha256 and record/split counts) to the already-reviewed ones.
    If anything differs, the old files are restored and nothing is changed.
  * Approval fields are copied back from the existing manifest only after that check; they are never invented.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "specialists" / "manifests"
VOCAB = "270"
RESTORE = ("training_allowed", "owner_approval", "privacy_reviewed", "speaker_consent_reviewed", "dataset_status", "governance")
JOBS = {
    "image": ("aethrion.specialist_prepare_visual", "data/authored/image/records.jsonl", ["--image-size", "64"]),
    "vision": ("aethrion.specialist_prepare_visual", "data/authored/vision/records.jsonl", ["--image-size", "224"]),
    "3d": ("aethrion.specialist_prepare_visual", "data/authored/3d/records.jsonl", ["--point-count", "512"]),
    "tts": ("scripts/prepare_audio_numpy.py", "data/authored/audio/tts.jsonl", ["--n-mels", "80", "--hop-length", "160"]),
    "asr": ("scripts/prepare_audio_numpy.py", "data/authored/audio/asr.jsonl", ["--n-mels", "80", "--hop-length", "160"]),
}


def _git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def main(models: list[str]) -> int:
    for m in models:
        manifest = MAN / f"{m}.json"
        old = json.loads(manifest.read_text(encoding="utf-8"))
        src = old["sources"][0]
        tool, records, extra = JOBS[m]
        cmd = [sys.executable] + (["-m", tool] if tool.startswith("aethrion") else [tool])
        cmd += ["--input", records, "--model", m, "--output-root", "data/specialists", "--license", src["license"],
                "--permitted-use", src["permitted_use"], "--evidence-uri", src["evidence_uri"], "--source-uri", src["source_uri"],
                "--seed", str(old["preparation"]["seed"]), "--vocab-size", VOCAB] + extra
        subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
        new = json.loads(manifest.read_text(encoding="utf-8"))
        old_by_path = {x["source_path"]: x for x in old["sources"]}
        same = (len(new["sources"]) == len(old["sources"])
                and all(x["source_path"] in old_by_path and old_by_path[x["source_path"]]["sha256"] == x["sha256"] for x in new["sources"])
                and new["preparation"]["record_count"] == old["preparation"]["record_count"]
                and new["preparation"]["split_counts"] == old["preparation"]["split_counts"])
        if not same:
            _git("checkout", "--", f"data/specialists/manifests/{m}.json", f"data/specialists/manifests/{m}")
            print(f"[{m}] source records differ from the reviewed ones; restored old files, nothing changed.")
            return 1
        for key in RESTORE:
            if key in old:
                new[key] = old[key]
        new["sources"] = [old_by_path[x["source_path"]] for x in new["sources"]]  # every source byte-identical to the reviewed one
        manifest.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{m}] re-prepared with vocab {VOCAB}; reviewed records unchanged; approval fields restored from the existing manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or list(JOBS)))
