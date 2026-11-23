"""Resumable training on the CI runner itself (CPU) -- no ZeroGPU quota needed.

Uses the same gated path as the Space: artifact pull -> data-gate preflight/train_family -> verified
checkpoint -> push to the private artifact repo. Sessions pause cleanly and resume. Completion is NOT a
release. Env: HF_TOKEN (required); AETHRION_BASE_AUDIT_HMAC_KEY (needed only for chat)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ["code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document", "chat"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--session-timeout", type=int, default=1500)
    ap.add_argument("--deadline-minutes", type=int, default=330)
    a = ap.parse_args()
    import time
    t0 = time.time()
    from huggingface_hub import HfApi
    from aethrion.space_session import run_session
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing", file=sys.stderr); return 2
    api = HfApi(token=token)
    os.environ.setdefault("AETHRION_ARTIFACT_REPO", f"{api.whoami()['name']}/aethrion-artifacts")
    summary = {}
    for model in a.models:
        state = {"ok": False, "complete": False, "step": None, "rounds": 0}
        failures = 0
        for _ in range(a.max_rounds):
            if time.time() - t0 > a.deadline_minutes * 60:
                print(f"[{model}] stopping: job deadline reached (resume by running again)", flush=True); break
            state["rounds"] += 1
            try:
                res = run_session(model, a.steps, root=ROOT, api=api, prepare_chat=True, timeout=a.session_timeout)
            except Exception as exc:
                print(f"[{model}] session error: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                failures += 1
                if failures >= 2: break
                continue
            prog = res.get("progress") or {}
            print(f"[{model}] round {state['rounds']} trainer_ok={res.get('trainer_ok')} progress={prog} "
                  f"pulled={len(res.get('pulled', []))} pushed={len(res.get('pushed', []))}", flush=True)
            if not res.get("trainer_ok"):
                print(f"[{model}] trainer tail: {str(res.get('trainer_tail'))[-400:]}", flush=True)
                failures += 1
                if failures >= 2: break
                continue
            failures = 0
            state.update(ok=True, step=prog.get("step"), complete=bool(prog.get("complete")))
            if state["complete"]: break
        summary[model] = state
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0 if all(v["ok"] for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
