"""Turn the owner's release decisions into release-evaluation records. Never approves on its own.

Inputs
  --decisions   JSON written by the owner: {"owner": "Name", "models": {"safety": {"karar": "yayin",
                "esikler": {"false_negative_rate": 0.15, "calibration_ece": 0.1}}, "code": {"karar": "hayir"}}}
  --audit       aethrion-specialist-evaluation.json (artifact of the `audit` workflow run)
  --release-dir folder holding <model>_evaluation.json templates (default: release)

A model is marked approved_for_release only if the owner wrote karar=yayin AND every required benchmark
has a MEASURED value in the audit file that meets the owner's threshold. Anything unmeasured stays pending
and is reported; there is no override. Writes a human review note per model and records its sha256."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

# benchmark name -> where its measured value lives in the audit entry's evaluation report
SOURCES = {
    "safety": {"calibration_ece": ("task_metrics", "calibration_ece"), "false_negative_rate": ("task_metrics", "false_negative_rate")},
    "research": {"recall_at_5": ("task_metrics", "recall_at_5")},
    "vision": {"recall_at_1": ("task_metrics", "accuracy")},
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(decisions: dict, audit: dict, release_dir: Path, now: str | None = None) -> dict[str, str]:
    owner = str(decisions.get("owner", "")).strip()
    if not owner:
        raise SystemExit("decisions.json must contain a non-empty 'owner'.")
    now = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    by_model = {e["model"]: e for e in audit.get("models", audit if isinstance(audit, list) else [])}
    result: dict[str, str] = {}
    for model, choice in decisions.get("models", {}).items():
        path = release_dir / f"{model}_evaluation.json"
        if not path.is_file():
            result[model] = "template missing"
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if choice.get("karar") != "yayin":
            result[model] = "owner decision: not released"
            continue
        evaluation = (by_model.get(model) or {}).get("evaluation") or {}
        problems = []
        for name, bench in report["benchmarks"].items():
            where = SOURCES.get(model, {}).get(name)
            value = evaluation
            for key in (where or ()):
                value = value.get(key) if isinstance(value, dict) else None
            threshold = (choice.get("esikler") or {}).get(name)
            if where is None or not isinstance(value, (int, float)):
                problems.append(f"{name}: no measurement")
                continue
            if not isinstance(threshold, (int, float)):
                problems.append(f"{name}: owner threshold missing")
                continue
            ok = value <= threshold if bench["direction"] == "min" else value >= threshold
            bench.update(metric=f"{'.'.join(where)} (audit, held-out synthetic data)", value=float(value),
                         threshold=float(threshold), passed=bool(ok),
                         evidence=f"aethrion-specialist-evaluation.json model={model}")
            if not ok:
                problems.append(f"{name}: {value:.4f} does not meet owner threshold {threshold}")
        if problems:
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result[model] = "NOT approved: " + "; ".join(problems)
            continue
        note = path.with_name(f"{model}_owner_review.txt")
        note.write_text(
            f"Owner release decision for {model}\nOwner: {owner}\nDate: {now}\n"
            f"Measured on held-out SYNTHETIC data only; real-world quality not verified.\n"
            + "".join(f"{n}: {b['value']} (threshold {b['threshold']})\n" for n, b in report["benchmarks"].items()),
            encoding="utf-8")
        report.update(status="approved_for_release", quality_gate_passed=True, safety_gate_passed=True,
                      human_reviewed=True, reviewer=owner, reviewed_at=now,
                      approval_evidence_path=str(note), approval_evidence_sha256=_digest(note))
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result[model] = "approved_for_release (record written; registry still verifies the rest)"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--release-dir", default="release")
    a = ap.parse_args()
    out = apply(json.loads(Path(a.decisions).read_text(encoding="utf-8")),
                json.loads(Path(a.audit).read_text(encoding="utf-8")), Path(a.release_dir))
    for model, text in out.items():
        print(f"{model}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
