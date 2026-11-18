import json
import tempfile
import unittest
from pathlib import Path

from aethrion.session_journal import append_session_event, audit_session_journal, audit_session_artifacts


class SessionJournalIntegrityTests(unittest.TestCase):
    def test_hash_chain_and_interrupted_recovery_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            events = [
                {"event": "session_started", "session_id": "s-1"},
                {"event": "model_attempt_started", "session_id": "s-1",
                 "attempt_id": "a-1", "model_id": "code"},
                {"event": "model_started", "session_id": "s-1",
                 "attempt_id": "a-1", "model_id": "code"},
                {"event": "session_interrupted", "session_id": "s-1"},
            ]
            for event in events:
                append_session_event(path, event)
            result = audit_session_journal(path, "s-1")
            self.assertTrue(result["valid"])
            self.assertEqual(result["event_count"], 4)
            self.assertEqual(result["recovery"], "checkpoint_verification_required")
            self.assertEqual(result["active_attempt_ids"], ["a-1"])

    def test_tampered_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            append_session_event(path, {"event": "session_started", "session_id": "s-1"})
            raw = path.read_text(encoding="utf-8").replace("session_started", "session_completed")
            path.write_text(raw, encoding="utf-8")
            result = audit_session_journal(path, "s-1")
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "hash_chain_mismatch")

    def test_truncated_final_line_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            append_session_event(path, {"event": "session_started", "session_id": "s-1"})
            with path.open("ab") as stream:
                stream.write(b'{"event":"session_interrupted"')
            result = audit_session_journal(path, "s-1")
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "truncated_final_line")

    def test_duplicate_attempt_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            append_session_event(path, {"event": "session_started", "session_id": "s-1"})
            append_session_event(path, {"event": "model_attempt_started", "session_id": "s-1",
                                        "attempt_id": "a-1", "model_id": "code"})
            # A duplicate cannot be appended to an already valid chain.
            with self.assertRaises(ValueError):
                append_session_event(path, {"event": "model_attempt_started", "session_id": "s-1",
                                            "attempt_id": "a-1", "model_id": "vision"})

    def test_finish_without_process_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            append_session_event(path, {"event": "session_started", "session_id": "s-1"})
            append_session_event(path, {"event": "model_attempt_started", "session_id": "s-1",
                                        "attempt_id": "a-1", "model_id": "code"})
            with self.assertRaises(ValueError):
                append_session_event(path, {"event": "model_finished", "session_id": "s-1",
                                            "attempt_id": "a-1", "model_id": "code",
                                            "status": "training_completed_not_released"})

    def test_snapshot_must_match_verified_journal_head_before_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            journal = root / "session-1.jsonl"
            append_session_event(journal, {"event": "session_started", "session_id": "session-1"})
            audit = audit_session_journal(journal, "session-1")
            snapshot = {
                "session_id": "session-1", "status": "interrupted",
                "journal_event_count": audit["event_count"],
                "journal_last_event_sha256": audit["last_event_sha256"],
            }
            (root / "session-1.json").write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertTrue(audit_session_artifacts(root, "session-1")["valid"])
            snapshot["journal_last_event_sha256"] = "f" * 64
            (root / "session-1.json").write_text(json.dumps(snapshot), encoding="utf-8")
            result = audit_session_artifacts(root, "session-1")
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "snapshot_journal_hash_mismatch")

    def test_legacy_unhashed_journal_is_not_claimed_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.jsonl"
            path.write_text(json.dumps({"event": "session_started", "session_id": "old"}) + "\\n",
                            encoding="utf-8")
            result = audit_session_journal(path, "old")
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "hash_chain_mismatch")

    def test_recovery_reconstructs_running_model_without_auto_resuming(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            journal = root / "session-2.jsonl"
            append_session_event(journal, {"event": "session_started", "session_id": "session-2"})
            append_session_event(journal, {"event": "model_attempt_started", "session_id": "session-2",
                                           "attempt_id": "attempt-2", "model_id": "code"})
            append_session_event(journal, {"event": "model_started", "session_id": "session-2",
                                           "attempt_id": "attempt-2", "model_id": "code"})
            audit = audit_session_journal(journal, "session-2")
            snapshot = {
                "session_id": "session-2", "status": "running", "current_model": "code",
                "execute_requested": True,
                "journal_event_count": audit["event_count"],
                "journal_last_event_sha256": audit["last_event_sha256"],
                "models": [{"model_id": "code", "status": "running",
                            "attempt_id": "attempt-2", "process_started": True}],
            }
            (root / "session-2.json").write_text(json.dumps(snapshot), encoding="utf-8")
            result = audit_session_artifacts(root, "session-2")
            self.assertTrue(result["valid"])
            rebuilt = result["reconstructed_session"]
            self.assertFalse(rebuilt["automatic_resume"])
            self.assertEqual(rebuilt["models"][0]["recovery_action"],
                             "checkpoint_verification_required")


if __name__ == "__main__":
    unittest.main()
