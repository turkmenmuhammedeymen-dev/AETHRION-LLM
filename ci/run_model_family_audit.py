"""Run and persist a fail-closed audit of all eleven native AETHRION models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aethrion.family_audit import EXPECTED_MODELS, audit_model_family


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report = audit_model_family(root)

    if report.get("model_count") != len(EXPECTED_MODELS):
        raise SystemExit("Family audit did not inspect exactly eleven models.")
    if report.get("execution_policy") != "local_only":
        raise SystemExit("Family audit detected a non-local execution policy.")
    if report.get("training_started") is not False:
        raise SystemExit("Readiness audit must never start training.")
    if report.get("automatic_approval") is not False:
        raise SystemExit("Readiness audit must never approve data or releases.")
    if report.get("registry_error") is not None:
        raise SystemExit(f"Native registry validation failed: {report['registry_error']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)

    print(json.dumps({
        "model_count": report["model_count"],
        "trained_count": report["trained_count"],
        "training_ready_count": report["training_ready_count"],
        "release_ready_count": report["release_ready_count"],
        "training_started": False,
        "automatic_approval": False,
        "report": str(output),
        "models": [
            {
                "model_id": row["model_id"],
                "registry_status": row["verified_registry_status"],
                "training_ready": row["training_ready"],
                "release_ready": row["release_ready"],
                "training_block_reason": row["training_block_reason"],
            }
            for row in report["models"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
