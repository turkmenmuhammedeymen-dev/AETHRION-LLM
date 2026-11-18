"""Tamper-evident JSONL journal and fail-closed family-session recovery audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_ZERO_HASH = "0" * 64
_CHAIN_FIELDS = {"previous_event_sha256", "event_sha256"}


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def audit_session_journal(path: str | Path, expected_session_id: str | None = None) -> dict[str, Any]:
    """Verify the complete hash chain and lifecycle ordering without mutating files."""
    journal = Path(path)
    if not journal.is_file():
        return {"valid": False, "reason": "journal_missing", "event_count": 0,
                "last_event_sha256": _ZERO_HASH, "attempts": {}, "recovery": "manual_review_required"}
    previous = _ZERO_HASH
    attempts: dict[str, dict[str, Any]] = {}
    event_count = 0
    session_id = expected_session_id
    last_event = None
    try:
        with journal.open("rb") as stream:
            for raw_line in stream:
                if not raw_line.endswith(b"\n"):
                    return {"valid": False, "reason": "truncated_final_line", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    return {"valid": False, "reason": "invalid_json_line", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                if not isinstance(event, dict):
                    return {"valid": False, "reason": "event_not_object", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                supplied_previous = event.get("previous_event_sha256")
                supplied_hash = event.get("event_sha256")
                payload = {key: value for key, value in event.items() if key not in _CHAIN_FIELDS}
                calculated = hashlib.sha256(previous.encode("ascii") + b"\n" + _canonical(payload)).hexdigest()
                if supplied_previous != previous or supplied_hash != calculated:
                    return {"valid": False, "reason": "hash_chain_mismatch", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                current_session = event.get("session_id")
                if not isinstance(current_session, str) or not current_session:
                    return {"valid": False, "reason": "session_id_missing", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                if session_id is None:
                    session_id = current_session
                if current_session != session_id:
                    return {"valid": False, "reason": "session_id_mismatch", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                kind = event.get("event")
                if event_count == 0 and kind != "session_started":
                    return {"valid": False, "reason": "session_start_event_missing",
                            "event_count": event_count, "last_event_sha256": previous,
                            "attempts": attempts, "recovery": "manual_review_required"}
                if event_count > 0 and kind == "session_started":
                    return {"valid": False, "reason": "duplicate_session_start",
                            "event_count": event_count, "last_event_sha256": previous,
                            "attempts": attempts, "recovery": "manual_review_required"}
                if last_event in {"session_interrupted", "session_completed"}:
                    return {"valid": False, "reason": "event_after_terminal_session",
                            "event_count": event_count, "last_event_sha256": previous,
                            "attempts": attempts, "recovery": "manual_review_required"}
                attempt_id = event.get("attempt_id")
                if kind == "model_attempt_started":
                    if not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempts:
                        return {"valid": False, "reason": "duplicate_or_missing_attempt_id",
                                "event_count": event_count, "last_event_sha256": previous,
                                "attempts": attempts, "recovery": "manual_review_required"}
                    if not isinstance(event.get("model_id"), str):
                        return {"valid": False, "reason": "attempt_model_missing", "event_count": event_count,
                                "last_event_sha256": previous, "attempts": attempts,
                                "recovery": "manual_review_required"}
                    attempts[attempt_id] = {"model_id": event["model_id"], "state": "attempt_prepared"}
                elif kind in {"model_started", "model_finished"}:
                    record = attempts.get(attempt_id)
                    if record is None or record["model_id"] != event.get("model_id"):
                        return {"valid": False, "reason": "orphan_attempt_event", "event_count": event_count,
                                "last_event_sha256": previous, "attempts": attempts,
                                "recovery": "manual_review_required"}
                    if kind == "model_started":
                        if record["state"] != "attempt_prepared":
                            return {"valid": False, "reason": "duplicate_model_start", "event_count": event_count,
                                    "last_event_sha256": previous, "attempts": attempts,
                                    "recovery": "manual_review_required"}
                        record["state"] = "process_running"
                    else:
                        if record["state"] != "process_running":
                            return {"valid": False, "reason": "finish_without_start", "event_count": event_count,
                                    "last_event_sha256": previous, "attempts": attempts,
                                    "recovery": "manual_review_required"}
                        record["state"] = "terminal_recorded"
                        record["terminal_status"] = event.get("status")
                        record["checkpoint_generation"] = event.get("checkpoint_generation")
                        record["checkpoint_sha256"] = event.get("checkpoint_sha256")
                        record["report_created_at_ns"] = event.get("report_created_at_ns")
                elif kind not in {
                    "session_started", "model_blocked", "session_interrupted", "session_completed"
                }:
                    return {"valid": False, "reason": "unknown_event_type", "event_count": event_count,
                            "last_event_sha256": previous, "attempts": attempts,
                            "recovery": "manual_review_required"}
                previous = supplied_hash
                event_count += 1
                last_event = kind
    except OSError:
        return {"valid": False, "reason": "journal_unreadable", "event_count": event_count,
                "last_event_sha256": previous, "attempts": attempts,
                "recovery": "manual_review_required"}

    active = [key for key, value in attempts.items() if value["state"] == "process_running"]
    prepared = [key for key, value in attempts.items() if value["state"] == "attempt_prepared"]
    if active:
        recovery = "checkpoint_verification_required"
    elif prepared:
        recovery = "safe_to_requeue_after_preflight"
    elif last_event == "session_interrupted":
        recovery = "safe_to_resume_after_preflight"
    elif last_event == "session_completed":
        recovery = "session_terminal"
    else:
        recovery = "inspect_snapshot_and_preflight"
    return {
        "valid": True, "reason": None, "session_id": session_id,
        "event_count": event_count, "last_event_sha256": previous,
        "attempts": attempts, "active_attempt_ids": active,
        "prepared_attempt_ids": prepared, "last_event": last_event,
        "recovery": recovery,
    }


def append_session_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append a chained event; refuse to extend an invalid or legacy journal."""
    journal = Path(path)
    current = audit_session_journal(journal, event.get("session_id")) if journal.exists() else {
        "valid": True, "event_count": 0, "last_event_sha256": _ZERO_HASH,
        "attempts": {}, "recovery": "new_session",
    }
    if not current["valid"]:
        raise ValueError(f"Refusing to append to invalid session journal: {current['reason']}")
    kind = event.get("event")
    if current["event_count"] == 0 and kind != "session_started":
        raise ValueError("session_started must be the first journal event.")
    if current.get("last_event") in {"session_interrupted", "session_completed"}:
        raise ValueError("Cannot append after a terminal session event.")
    attempt_id = event.get("attempt_id")
    record = current.get("attempts", {}).get(attempt_id)
    if kind == "session_started":
        if current["event_count"] != 0:
            raise ValueError("session_started must be the first journal event.")
    elif kind == "model_attempt_started":
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in current.get("attempts", {}):
            raise ValueError("Attempt ID is missing or already used in this session.")
        if not isinstance(event.get("model_id"), str) or not event["model_id"]:
            raise ValueError("Attempt model_id is required.")
    elif kind == "model_started":
        if record is None or record["state"] != "attempt_prepared" or record["model_id"] != event.get("model_id"):
            raise ValueError("model_started requires its unique prepared attempt.")
    elif kind == "model_finished":
        if record is None or record["state"] != "process_running" or record["model_id"] != event.get("model_id"):
            raise ValueError("model_finished requires a matching started process.")
    elif kind not in {"model_blocked", "session_interrupted", "session_completed"}:
        raise ValueError("Unknown or unsupported session event type.")
    payload = dict(event)
    if _CHAIN_FIELDS & set(payload):
        raise ValueError("Event payload cannot supply journal hash fields.")
    previous = current["last_event_sha256"]
    chained = {**payload, "previous_event_sha256": previous}
    chained["event_sha256"] = hashlib.sha256(
        previous.encode("ascii") + b"\n" + _canonical(payload)
    ).hexdigest()
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("ab") as stream:
        stream.write(_canonical(chained) + b"\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return {"event_count": current["event_count"] + 1,
            "last_event_sha256": chained["event_sha256"]}


def audit_session_artifacts(session_dir: str | Path, session_id: str) -> dict[str, Any]:
    """Cross-check the session snapshot against its journal before recovery."""
    directory = Path(session_dir)
    journal_result = audit_session_journal(directory / f"{session_id}.jsonl", session_id)
    if not journal_result.get("valid"):
        return journal_result
    snapshot_path = directory / f"{session_id}.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {**journal_result, "valid": False, "reason": "snapshot_missing_or_invalid",
                "recovery": "manual_review_required"}
    if not isinstance(snapshot, dict) or snapshot.get("session_id") != session_id:
        return {**journal_result, "valid": False, "reason": "snapshot_session_mismatch",
                "recovery": "manual_review_required"}
    snapshot_count = snapshot.get("journal_event_count")
    snapshot_hash = snapshot.get("journal_last_event_sha256")
    if snapshot_count is not None and snapshot_count != journal_result["event_count"]:
        return {**journal_result, "valid": False, "reason": "snapshot_event_count_mismatch",
                "recovery": "manual_review_required"}
    if snapshot_hash is not None and snapshot_hash != journal_result["last_event_sha256"]:
        return {**journal_result, "valid": False, "reason": "snapshot_journal_hash_mismatch",
                "recovery": "manual_review_required"}
    if snapshot_count is None or snapshot_hash is None:
        return {**journal_result, "valid": False, "reason": "legacy_snapshot_unverifiable",
                "recovery": "manual_review_required"}
    reconstructed_rows = []
    snapshot_rows = snapshot.get("models", [])
    if not isinstance(snapshot_rows, list):
        return {**journal_result, "valid": False, "reason": "snapshot_models_invalid",
                "recovery": "manual_review_required"}
    attempts_by_model = {}
    for attempt_id, attempt in journal_result.get("attempts", {}).items():
        attempts_by_model.setdefault(attempt.get("model_id"), []).append((attempt_id, attempt))
    for row in snapshot_rows:
        if not isinstance(row, dict) or not isinstance(row.get("model_id"), str):
            return {**journal_result, "valid": False, "reason": "snapshot_model_row_invalid",
                    "recovery": "manual_review_required"}
        rebuilt = dict(row)
        model_attempts = attempts_by_model.get(row["model_id"], [])
        if len(model_attempts) > 1:
            return {**journal_result, "valid": False, "reason": "multiple_attempts_for_model",
                    "recovery": "manual_review_required"}
        if model_attempts:
            attempt_id, attempt = model_attempts[0]
            if row.get("attempt_id") not in (None, attempt_id):
                return {**journal_result, "valid": False, "reason": "snapshot_attempt_mismatch",
                        "recovery": "manual_review_required"}
            rebuilt["attempt_id"] = attempt_id
            if attempt["state"] == "process_running":
                rebuilt["recovery_action"] = "checkpoint_verification_required"
            elif attempt["state"] == "attempt_prepared":
                rebuilt["recovery_action"] = "safe_to_requeue_after_preflight"
            else:
                rebuilt["recovery_action"] = "terminal_event_recorded"
                rebuilt["journal_terminal_status"] = attempt.get("terminal_status")
        reconstructed_rows.append(rebuilt)
    return {
        **journal_result,
        "snapshot_status": snapshot.get("status"),
        "snapshot_verified": True,
        "reconstructed_session": {
            "session_id": session_id,
            "status": snapshot.get("status"),
            "current_model": snapshot.get("current_model"),
            "models": reconstructed_rows,
            "execute_requested": snapshot.get("execute_requested") is True,
            "released_count": 0,
            "automatic_resume": False,
        },
    }
