"""Read-only audit + held-out evaluation of the specialist checkpoints in the private artifact repo.

Pulls each model's checkpoint into a scratch directory, verifies sha256 sidecars, loads with
weights_only, checks finite weights/step/dataset binding and runs aethrion.specialist_eval.
Writes a JSON report. Never trains, never pushes, never changes a release status.
Exit code is non-zero only when an integrity check fails (missing/corrupt/mismatched checkpoint)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ["code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document"]


def audit_model(model, scratch, api):
    from aethrion import artifact_sync
    from aethrion.specialist_eval import evaluate_specialist
    cfg_path = ROOT / "configs" / "specialists" / f"{model}.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = Path(scratch) / model
    pulled = artifact_sync.pull_model_artifacts(model, out, api=api)
    entry = {"model": model, "pulled": pulled, "integrity_ok": False}
    if "checkpoint_last.pt" not in pulled or "checkpoint_last.pt.sha256" not in pulled:
        entry["error"] = "checkpoint or sha256 sidecar missing in artifact repo"
        return entry
    try:
        progress = json.loads((out / "training_report.json").read_text(encoding="utf-8"))
        entry["training_report"] = {k: progress.get(k) for k in ("status", "step", "steps", "validation_loss", "held_out_test_loss")}
    except Exception:
        entry["training_report"] = None
    manifest = ROOT / "data" / "specialists" / "manifests" / f"{model}.json"
    try:
        entry["evaluation"] = evaluate_specialist(config, manifest, out / "checkpoint_last.pt")
        entry["integrity_ok"] = True
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        entry["trace"] = traceback.format_exc()[-600:]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/tmp/aethrion-specialist-evaluation.json")
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing (CircleCI context aethrion-deploy).", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    owner = api.whoami().get("name")
    os.environ.setdefault("AETHRION_ARTIFACT_REPO", f"{owner}/aethrion-artifacts")
    scratch = Path("/tmp/aethrion-audit")
    results = []
    for m in a.models:
        r = audit_model(m, scratch, api)
        ev = r.get("evaluation", {})
        print(f"[{m}] integrity_ok={r['integrity_ok']} step={ev.get('checkpoint_step')}/{ev.get('max_steps')} "
              f"test_loss={ev.get('test_loss')} baseline={ev.get('untrained_baseline_test_loss')} "
              f"learning={ev.get('evidence_of_learning')} task={ev.get('task_metrics')} err={r.get('error')}", flush=True)
        results.append(r)
    report = {"format": "aethrion-specialist-evaluation-v1", "released_models": [], "models": results,
              "note": "Evaluation on held-out synthetic first-party data; not a release decision."}
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(r["integrity_ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
