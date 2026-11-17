"""Create a pending human release-review template; never auto-approve a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aethrion.native_registry import REQUIRED_BENCHMARKS


class ReleaseReviewError(ValueError):
    pass


def create_release_evaluation_template(
    model_id: str,
    training_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a deliberately unapproved benchmark/review report for a completed training run."""
    if model_id not in REQUIRED_BENCHMARKS:
        raise ReleaseReviewError(f"Unknown AETHRION model ID: {model_id}")
    training_path = Path(training_report_path).resolve()
    try:
        training = json.loads(training_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseReviewError("Training report is missing or invalid JSON.") from exc
    if training.get("model_id") != model_id or training.get("status") not in {
        "training_complete_not_released", "instruction_tuning_complete_not_released"
    }:
        raise ReleaseReviewError("Training report does not prove a completed run for this model.")
    dataset_hash = training.get("dataset_sha256")
    checkpoint_hash = training.get("checkpoint_sha256")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ReleaseReviewError("Training report lacks a dataset SHA-256.")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ReleaseReviewError("Training report lacks a checkpoint SHA-256.")

    report = {
        "format": "aethrion-release-evaluation-v1",
        "model_id": model_id,
        "dataset_sha256": dataset_hash,
        "checkpoint_sha256": checkpoint_hash,
        "status": "pending_review",
        "quality_gate_passed": False,
        "safety_gate_passed": False,
        "human_reviewed": False,
        "reviewer": "",
        "reviewed_at": "",
        "approval_evidence_path": "",
        "approval_evidence_sha256": "",
        "benchmarks": {
            name: {
                "metric": "",
                "value": None,
                "threshold": None,
                "passed": False,
                "evidence": "",
            }
            for name in sorted(REQUIRED_BENCHMARKS[model_id])
        },
        "review_instructions": (
            "Measure each required benchmark on a contamination-checked independent held-out "
            "dataset. Add the exact metric, measured value, acceptance threshold and evidence. "
            "A qualified human reviewer must verify quality, privacy, safety, limitations and "
            "model-card accuracy. Keep this report pending_review until that review is complete. "
            "Never mark a benchmark passed without a real measurement."
        ),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Create an unapproved AETHRION release evaluation template.")
    parser.add_argument("--model-id", required=True, choices=sorted(REQUIRED_BENCHMARKS))
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = create_release_evaluation_template(args.model_id, args.training_report, args.output)
    print(json.dumps({
        "model_id": report["model_id"],
        "status": report["status"],
        "required_benchmarks": sorted(report["benchmarks"]),
        "automatic_approval": False,
        "output": str(Path(args.output)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
