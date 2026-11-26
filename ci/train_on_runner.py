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
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--session-timeout", type=int, default=1500)
    ap.add_argument("--deadline-minutes", type=int, default=330)
    ap.add_argument("--report", default=None, help="Optional JSON summary report path.")
    ap.add_argument("--fresh", action="store_true", help="delete the model's old remote checkpoints first (dataset changed)")
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
        if a.fresh:
            try:
                api.delete_folder(path_in_repo=model, repo_id=os.environ["AETHRION_ARTIFACT_REPO"], repo_type="model",
                                  commit_message=f"fresh retrain: remove stale {model} checkpoints")
                print(f"[{model}] fresh: old remote checkpoints removed", flush=True)
            except Exception as exc:
                print(f"[{model}] fresh: nothing removed ({str(exc)[:120]})", flush=True)
        state = {"ok": False, "complete": False, "step": None, "rounds": 0}
        failures = 0
        waits = 0
        for _ in range(a.max_rounds + 8):
            if time.time() - t0 > a.deadline_minutes * 60:
                print(f"[{model}] stopping: job deadline reached (resume by running again)", flush=True); break
            if state["rounds"] >= a.max_rounds: break
            state["rounds"] += 1
            try:
                res = run_session(model, a.steps, root=ROOT, api=api, prepare_chat=True, timeout=a.session_timeout)
            except Exception as exc:
                if "429" in str(exc) and waits < 8:  # Hub commit rate limit: wait, then retry the same round
                    waits += 1; state["rounds"] -= 1
                    print(f"[{model}] Hub rate limit (429); waiting 10 min (wait {waits}/8)", flush=True)
                    time.sleep(600); continue
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
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("SUMMARY", rendered, flush=True)
    if a.report:
        report_path = Path(a.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(report_path) + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(report_path)
    return 0 if all(v["ok"] for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
