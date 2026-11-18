"""Human-review attestations required before any AETHRION dataset can train.

This validates a documented review record; it cannot independently prove authorship,
copyright ownership, consent, or reviewer identity. Evidence must be retained and
checked by a responsible human before an approval is entered.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

SCHEMA = "aethrion-human-review-v1"
AUDIO_MODELS = {"asr", "tts", "vocoder"}


class GovernanceError(ValueError):
    """Missing, malformed, or incomplete dataset governance evidence."""


def review_template(model_id: str) -> dict[str, Any]:
    """Return a deliberately unapproved review record for a human to complete."""
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "approval_status": "pending",
        "author_role": "",
        "authorship_attested": False,
        "rights_holder_role": "",
        "rights_attested": False,
        "source_rights_reviewed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "reviewer_role": "",
        "reviewed_at": "",
        "evidence_uri": "",
        "speaker_consent_reviewed": False,
        "safety_reviewed": False,
    }


def validate_governance_record(record: Any, *, model_id: str) -> dict[str, Any]:
    """Require explicit, timestamped, role-attributed human review for a model dataset."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise GovernanceError("Training blocked: aethrion-human-review-v1 governance record is required.")
    if record.get("model_id") != model_id:
        raise GovernanceError("Training blocked: governance model_id does not match the dataset.")
    if record.get("approval_status") != "approved":
        raise GovernanceError("Training blocked: human governance approval is not approved.")

    for key in ("author_role", "rights_holder_role", "reviewer_role", "evidence_uri"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise GovernanceError(f"Training blocked: governance field {key} is required.")
    for key in (
        "authorship_attested", "rights_attested", "source_rights_reviewed",
        "privacy_reviewed", "quality_reviewed",
    ):
        if record.get(key) is not True:
            raise GovernanceError(f"Training blocked: governance review {key} must be true.")

    reviewed_at = record.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise GovernanceError("Training blocked: governance reviewed_at timestamp is required.")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GovernanceError("Training blocked: governance reviewed_at must be ISO-8601.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise GovernanceError("Training blocked: governance reviewed_at must include a timezone.")

    if model_id in AUDIO_MODELS and record.get("speaker_consent_reviewed") is not True:
        raise GovernanceError("Training blocked: explicit speaker-consent review is required for audio.")
    if model_id == "safety" and record.get("safety_reviewed") is not True:
        raise GovernanceError("Training blocked: SAFE data requires separate human safety review.")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an unapproved AETHRION human-review template.")
    parser.add_argument("--model", required=True, choices=(
        "base", "chat", "code", "vision", "image", "asr", "tts", "vocoder",
        "3d", "research", "document", "safety",
    ))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review_template(args.model), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "approval_status": "pending"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
