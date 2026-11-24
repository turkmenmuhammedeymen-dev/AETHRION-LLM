"""CHAT stage 2 on the CI runner: instruction-tune from the verified BASE checkpoint (CPU), never a release.

Steps: deterministic BASE tokenizer (scripts/prepare_base.py, needs AETHRION_BASE_AUDIT_HMAC_KEY) ->
pull BASE checkpoint from the private artifact repo (prefix `chat`) -> prepare the owner-attested CHAT
instruction dataset with that exact tokenizer -> attestation bound to the records hash -> instruction tuning
(response-only loss) -> push result under prefix `chat_instruct`. Sessions are not resumed (runs to completion)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=ROOT, check=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token or len(os.environ.get("AETHRION_BASE_AUDIT_HMAC_KEY", "").encode()) < 32:
        print("HF_TOKEN and AETHRION_BASE_AUDIT_HMAC_KEY (>=32 bytes) are required.", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    from aethrion import artifact_sync
    api = HfApi(token=token)
    os.environ.setdefault("AETHRION_ARTIFACT_REPO", f"{api.whoami()['name']}/aethrion-artifacts")
    sh(sys.executable, "scripts/prepare_base.py")
    pulled = artifact_sync.pull_model_artifacts("chat", ROOT / "artifacts", api=api)
    if "checkpoint_last.pt" not in pulled:
        print("No BASE checkpoint in the artifact repo; run train-runner for chat first.", file=sys.stderr)
        return 3
    from aethrion.checkpoint import load_checkpoint
    base = load_checkpoint(ROOT / "artifacts" / "checkpoint_last.pt", map_location="cpu")
    print("BASE checkpoint step:", base.get("step"), flush=True)
    if (ROOT / "data/specialists/manifests/chat.json").exists():
        print("chat manifest already present; refusing to overwrite", file=sys.stderr)
        return 4
    gov = json.loads((ROOT / "data/authored/chat/GOVERNANCE.json").read_text(encoding="utf-8"))
    sh(sys.executable, "-m", "aethrion.specialist_prepare", "--input", "data/authored/chat/records.jsonl",
       "--model", "chat", "--license", "First-party owner-authored", "--permitted-use", "model training permitted",
       "--evidence-uri", "https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/data/authored/chat/GOVERNANCE.json",
       "--source-uri", "owner-authored:aethrion-llm/data/authored/chat", "--tokenizer", "artifacts/tokenizer.json", "--max-length", "256")
    sh(sys.executable, "scripts/apply_owner_attestation.py", "chat", "--session", "01CgbDesHhneLFEzEj5NSZrK",
       "--statement", gov["owner_statement"])
    sh(sys.executable, "-m", "aethrion.instruction_tune", "--config", "configs/instruction_tuning.json",
       "--manifest", "data/specialists/manifests/chat.json", "--base-checkpoint", "artifacts/checkpoint_last.pt")
    out = ROOT / "artifacts" / "models" / "chat"
    report = json.loads((out / "instruction_tuning_report.json").read_text(encoding="utf-8"))
    print("REPORT", json.dumps({k: report.get(k) for k in ("status", "release_status", "steps", "validation_loss", "test_loss", "held_out_test_loss")}), flush=True)
    import shutil
    stage = ROOT / "artifacts" / "push_chat_instruct"
    shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
    for name in sorted(artifact_sync.ALLOWED_ARTIFACT_NAMES):
        if (out / name).is_file():
            shutil.copyfile(out / name, stage / name)
    sent = artifact_sync.push_model_artifacts("chat_instruct", stage, api=api)
    print("pushed", sent, flush=True)
    return 0 if str(report.get("status", "")).startswith("instruction_tuning_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
