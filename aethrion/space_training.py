"""Read-only Hugging Face Space view of the eleven native training gates."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_repository_root(space_file: str | Path) -> Path:
    """Locate the checkout containing both the trainer and model configs."""
    source = Path(space_file).resolve()
    start = source if source.is_dir() else source.parent
    for candidate in (start, *start.parents):
        if (candidate / "aethrion" / "train_family.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("AETHRION repository root with configs/ and aethrion/ was not found.")


def render_training_readiness(space_file: str | Path) -> str:
    """Run only preflight_model through run_family(execute=False); never train."""
    from aethrion.train_family import run_family

    root = resolve_repository_root(space_file)
    report: dict[str, Any] = run_family(root, execute=False)
    rows = [
        "### AETHRION-LLM · 11-model training readiness",
        f"- **Models checked:** {report['model_count']}",
        f"- **Data gates ready:** {report['ready_count']}",
        f"- **Blocked:** {report['blocked_count']}",
        "- **Training started:** NO",
        "- **Checkpoints written:** NO",
        "- **Training approval changed:** NO",
        "",
        "| Model | Gate | Details |",
        "|---|---|---|",
    ]
    for item in report.get("models", []):
        model_id = str(item.get("model_id", "unknown")).replace("|", "\\|")
        state = "READY — NOT STARTED" if item.get("ready") is True else "BLOCKED"
        detail = str(item.get("reason", "All required local inputs and approvals are present."))
        detail = " ".join(detail.split()).replace("|", "\\|")[:280]
        rows.append(f"| {model_id} | {state} | {detail} |")
    rows.extend([
        "",
        "**Important:** READY means only that the current local preflight passed. "
        "It does not mean the model is trained, evaluated, or released.",
        "**Execution policy:** this Space control is permanently preflight-only; "
        "training remains a separate, explicitly reviewed operation.",
    ])
    return "\n".join(rows)
