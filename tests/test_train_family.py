import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aethrion.train_family import MODEL_IDS, FamilyTrainingError, _resolve_output_dir, preflight_model, run_family


class FamilyTrainingOrchestratorTests(unittest.TestCase):
    def _empty_repo(self, root: Path):
        (root / "configs" / "specialists").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        for model_id in MODEL_IDS:
            if model_id == "chat":
                (root / "configs" / "small.json").write_text(
                    '{"data_path":"data/processed/corpus.txt",'
                    '"tokenizer_path":"artifacts/tokenizer.json","output_dir":"artifacts"}',
                    encoding="utf-8",
                )
                (root / "data" / "processed" / "corpus_manifest.json").write_text(
                    '{"training_allowed":false}', encoding="utf-8"
                )
            else:
                (root / "configs" / "specialists" / f"{model_id}.json").write_text(
                    '{"model_id":"' + model_id + '","output_dir":"artifacts/models/' + model_id + '"}',
                    encoding="utf-8",
                )

    def test_family_contains_exactly_eleven_models(self):
        self.assertEqual(len(MODEL_IDS), 11)
        self.assertEqual(len(set(MODEL_IDS)), 11)

    def test_chat_preflight_fails_closed_without_approved_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._empty_repo(root)
            with self.assertRaises(Exception):
                preflight_model(root, "chat")

    def test_all_missing_specialist_manifests_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._empty_repo(root)
            report = run_family(root)
            self.assertEqual(report["model_count"], 11)
            self.assertEqual(report["ready_count"], 0)
            self.assertEqual(report["blocked_count"], 11)
            self.assertFalse(report["training_started"])
            self.assertEqual(report["released_count"], 0)

    def test_execute_does_not_call_trainers_when_data_gates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._empty_repo(root)
            with patch("aethrion.train_family.subprocess.run") as run:
                report = run_family(root, execute=True)
            run.assert_not_called()
            self.assertEqual(report["blocked_count"], 11)
            self.assertFalse(report["training_started"])

    def test_unknown_or_duplicate_model_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_family(tmp, selected=["chat", "chat"])
            with self.assertRaises(ValueError):
                run_family(tmp, selected=["external-model"])

    def test_output_directory_must_be_explicit_and_non_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for invalid in ("", ".", "..", "../outside", str(root / "absolute"),
                            r"artifacts\\models\\code", "C:/outside", "artifacts/./code", "artifacts//code"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(FamilyTrainingError):
                        _resolve_output_dir(root, invalid)

    def test_output_directory_rejects_windows_and_ambiguous_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for invalid in (r"artifacts\\models\\code", "C:/outside", "artifacts/./code", "artifacts//code"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(FamilyTrainingError):
                        _resolve_output_dir(root, invalid)

    def test_empty_model_selection_does_not_fall_back_to_all_models(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FamilyTrainingError, "At least one model"):
                run_family(folder, selected=[])

    def test_output_directory_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder).resolve()
            (root / "artifacts").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(FamilyTrainingError, "escapes"):
                _resolve_output_dir(root, "artifacts")

    def test_base_training_input_paths_reject_windows_and_dot_paths(self):
        from aethrion.train_family import _resolve_repo_input
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for invalid in (r"C:\\outside.txt", r"..\\outside.txt", "./inside.txt", "data//corpus.txt"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(FamilyTrainingError):
                        _resolve_repo_input(root, invalid, "data_path")

    def test_base_training_input_paths_require_existing_regular_files(self):
        from aethrion.train_family import _resolve_repo_input
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with self.assertRaisesRegex(FamilyTrainingError, "regular file"):
                _resolve_repo_input(root, "missing.txt", "data_path")
            valid = root / "inside.txt"
            valid.write_text("local", encoding="utf-8")
            self.assertEqual(_resolve_repo_input(root, "inside.txt", "data_path"), valid)

    def test_base_training_input_paths_reject_absolute_and_parent_paths(self):
        from aethrion.train_family import _resolve_repo_input
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for field in ("data_path", "tokenizer_path"):
                for invalid in ("", "../outside.txt", str(root / "outside.txt")):
                    with self.subTest(field=field, invalid=invalid):
                        with self.assertRaises(FamilyTrainingError):
                            _resolve_repo_input(root, invalid, field)

    def test_base_training_input_paths_reject_symlink_escape(self):
        from aethrion.train_family import _resolve_repo_input
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder).resolve()
            (root / "external-data").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(FamilyTrainingError, "escapes"):
                _resolve_repo_input(root, "external-data/corpus.txt", "data_path")

    def test_base_training_input_paths_allow_repository_relative_files(self):
        from aethrion.train_family import _resolve_repo_input
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            corpus = root / "data/processed/corpus.txt"
            corpus.parent.mkdir(parents=True)
            corpus.write_text("approved local fixture", encoding="utf-8")
            self.assertEqual(_resolve_repo_input(root, "data/processed/corpus.txt", "data_path"), corpus)

    def test_valid_output_directory_is_resolved_inside_repo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            self.assertEqual(_resolve_output_dir(root, "artifacts/models/code"), root / "artifacts/models/code")

    def test_specialist_preflight_rejects_missing_output_dir_before_data_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_dir = root / "configs" / "specialists"
            config_dir.mkdir(parents=True)
            (config_dir / "code.json").write_text(
                '{"model_id":"code","model_config":{"vocab_size":256},"output_dir":""}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FamilyTrainingError, "output_dir"):
                preflight_model(root, "code")

    def test_specialist_preflight_rejects_mismatched_model_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_dir = root / "configs" / "specialists"
            config_dir.mkdir(parents=True)
            (config_dir / "code.json").write_text(
                '{"model_id":"vision","model_config":{"vocab_size":256},"output_dir":"artifacts/models/code"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FamilyTrainingError, "model_id"):
                preflight_model(root, "code")


    def test_execute_writes_durable_session_snapshot_and_event_journal(self):
        import json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {
                "model_id": "chat", "ready": True,
                "command": ["python", "-m", "aethrion.train"],
                "checkpoint": "artifacts/chat/checkpoint_last.pt",
                "output_dir": "artifacts/chat",
            }
            with patch("aethrion.train_family.preflight_model", return_value=plan), \
                 patch("aethrion.train_family.subprocess.Popen") as popen, \
                 patch("aethrion.train_family._training_outcome", return_value="training_completed_not_released"):
                popen.return_value.wait.return_value = 0
                popen.return_value.poll.return_value = 0
                report = run_family(root, selected=["chat"], execute=True)

            self.assertTrue(report["training_started"])
            self.assertEqual(report["models"][0]["status"], "training_completed_not_released")
            session_dir = root / "artifacts" / "training_sessions"
            snapshots = list(session_dir.glob("*.json"))
            journals = list(session_dir.glob("*.jsonl"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(len(journals), 1)
            snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
            events = [json.loads(line) for line in journals[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(snapshot["status"], "completed")
            self.assertEqual(snapshot["released_count"], 0)
            self.assertEqual(
                [event["event"] for event in events],
                ["session_started", "model_started", "model_finished", "session_completed"],
            )

    def test_keyboard_interrupt_persists_interrupted_family_session(self):
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {
                "model_id": "chat", "ready": True,
                "command": ["python", "-m", "aethrion.train"],
                "checkpoint": "artifacts/chat/checkpoint_last.pt",
                "output_dir": "artifacts/chat",
            }
            with patch("aethrion.train_family.preflight_model", return_value=plan), \
                 patch("aethrion.train_family.subprocess.Popen") as popen, \
                 patch("aethrion.train_family._training_outcome", return_value="training_completed_not_released"):
                popen.return_value.wait.side_effect = [KeyboardInterrupt, -2]
                popen.return_value.poll.return_value = None
                report = run_family(root, selected=["chat"], execute=True)

            popen.return_value.send_signal.assert_called_once()
            self.assertTrue(report["training_started"])

            self.assertTrue(report["interrupted"])
            self.assertEqual(report["models"][0]["status"], "training_interrupted_not_released")
            snapshot = next((root / "artifacts" / "training_sessions").glob("*.json"))
            state = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["released_count"], 0)

    def test_failed_process_spawn_is_blocked_not_reported_as_training_started(self):
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {
                "model_id": "chat", "ready": True,
                "command": ["python", "-m", "aethrion.train"],
                "checkpoint": "artifacts/chat/checkpoint_last.pt",
                "output_dir": "artifacts/chat",
            }
            with patch("aethrion.train_family.preflight_model", return_value=plan), \\
                 patch("aethrion.train_family.subprocess.Popen", side_effect=OSError("runner unavailable")):
                report = run_family(root, selected=["chat"], execute=True)

            self.assertFalse(report["training_started"])
            self.assertEqual(len(report["models"]), 1)
            self.assertEqual(report["models"][0]["status"], "blocked")
            journal = next((root / "artifacts" / "training_sessions").glob("*.jsonl"))
            events = [json.loads(line)["event"] for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("model_started", events)
            self.assertIn("model_blocked", events)

    def test_family_passes_resume_when_only_previous_checkpoint_exists(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = root / "artifacts" / "chat" / "checkpoint_last.pt.previous"
            previous.parent.mkdir(parents=True)
            previous.write_bytes(b"previous-generation")
            plan = {
                "model_id": "chat", "ready": True,
                "command": ["python", "-m", "aethrion.train"],
                "checkpoint": "artifacts/chat/checkpoint_last.pt",
                "output_dir": "artifacts/chat",
            }
            with patch("aethrion.train_family.preflight_model", return_value=plan), \\
                 patch("aethrion.train_family.subprocess.Popen") as popen, \\
                 patch("aethrion.train_family._training_outcome", return_value="training_interrupted_not_released"):
                popen.return_value.wait.return_value = 0
                popen.return_value.poll.return_value = 0
                run_family(root, selected=["chat"], execute=True)

            command = popen.call_args.args[0]
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--resume") + 1], "artifacts/chat/checkpoint_last.pt")

    def test_specialist_config_preflight_checks_architecture_and_training_fields(self):
        from aethrion.train_family import _validate_specialist_config

        valid = {
            "seed": 42, "max_steps": 10, "batch_size": 2,
            "learning_rate": 0.001, "weight_decay": 0.01,
            "grad_clip": 1.0, "eval_interval": 2, "save_interval": 2,
            "model_config": {
                "vocab_size": 2048, "n_mels": 80, "width": 128,
                "layers": 4, "heads": 4, "blank_id": 0,
            },
        }
        _validate_specialist_config("asr", valid)
        invalid = {**valid, "model_config": {**valid["model_config"]}}
        invalid["model_config"].pop("n_mels")
        with self.assertRaisesRegex(FamilyTrainingError, "n_mels"):
            _validate_specialist_config("asr", invalid)

    def test_specialist_config_preflight_rejects_invalid_optimizer_settings(self):
        from aethrion.train_family import _validate_specialist_config

        config = {
            "seed": 42, "max_steps": 10, "batch_size": 2,
            "learning_rate": 0, "weight_decay": 0.01,
            "grad_clip": 1.0, "eval_interval": 2, "save_interval": 2,
            "model_config": {
                "vocab_size": 2048, "block_size": 512, "n_layer": 4,
                "n_head": 4, "n_embd": 128,
            },
        }
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _validate_specialist_config("code", config)

    def test_preflight_only_does_not_create_training_session_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._empty_repo(root)
            report = run_family(root, execute=False)
            self.assertFalse(report["training_started"])
            self.assertFalse((root / "artifacts" / "training_sessions").exists())


    def _write_verified_report_fixture(self, root: Path, *, model_id="code", status="training_complete_not_released"):
        import hashlib
        import json

        output = root / "artifacts" / "models" / model_id
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint_last.pt"
        checkpoint.write_bytes(b"verified-checkpoint-fixture")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report = {
            "status": status, "release_status": "NOT_RELEASED", "model_id": model_id,
            "steps": 10, "dataset_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "manifest_sha256": "c" * 64, "checkpoint_sha256": digest,
            "validation_loss": 1.25, "held_out_test_loss": 1.5,
        }
        (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        return output, checkpoint, report

    def test_training_outcome_accepts_only_fresh_report_with_matching_checkpoint(self):
        from aethrion.train_family import _training_outcome
        import time

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, _ = self._write_verified_report_fixture(Path(folder))
            outcome = _training_outcome(
                output, "code", 0, checkpoint_path=checkpoint,
                started_at_ns=time.time_ns() - 1_000_000_000,
            )
            self.assertEqual(outcome, "training_completed_not_released")

    def test_training_outcome_rejects_wrong_model_and_missing_checkpoint(self):
        from aethrion.train_family import _training_outcome

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, _ = self._write_verified_report_fixture(Path(folder))
            self.assertEqual(
                _training_outcome(output, "vision", 0, checkpoint_path=checkpoint),
                "training_report_model_mismatch_not_released",
            )
            checkpoint.unlink()
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_checkpoint_missing_not_released",
            )

    def test_training_outcome_rejects_checkpoint_hash_mismatch(self):
        from aethrion.train_family import _training_outcome

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, report = self._write_verified_report_fixture(Path(folder))
            report["checkpoint_sha256"] = "0" * 64
            import json
            (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_checkpoint_hash_mismatch_not_released",
            )

    def test_training_outcome_rejects_missing_or_nonfinite_metrics(self):
        from aethrion.train_family import _training_outcome
        import json

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, report = self._write_verified_report_fixture(Path(folder))
            report["held_out_test_loss"] = float("nan")
            (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_report_metrics_invalid_not_released",
            )

    def test_training_outcome_rejects_stale_report(self):
        from aethrion.train_family import _training_outcome
        import time

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, _ = self._write_verified_report_fixture(Path(folder))
            self.assertEqual(
                _training_outcome(
                    output, "code", 0, checkpoint_path=checkpoint,
                    started_at_ns=time.time_ns() + 1_000_000_000,
                ),
                "training_report_stale_not_released",
            )


    def test_training_outcome_requires_fresh_model_matched_report_and_verified_checkpoint(self):
        import hashlib
        import json
        import time
        from aethrion.train_family import _training_outcome

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / "checkpoint_last.pt"
            checkpoint.write_bytes(b"verified-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = {
                "status": "training_complete_not_released",
                "release_status": "NOT_RELEASED",
                "model_id": "code",
                "steps": 12,
                "dataset_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "checkpoint_sha256": digest,
                "validation_loss": 0.25,
                "held_out_test_loss": 0.3,
            }
            report_path = output / "training_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            started = time.time_ns() - 1_000_000

            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint, started_at_ns=started),
                "training_completed_not_released",
            )

            wrong_model = {**report, "model_id": "vision"}
            report_path.write_text(json.dumps(wrong_model), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_report_model_mismatch_not_released",
            )

            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=output / "missing.pt"),
                "training_checkpoint_missing_not_released",
            )

            checkpoint.write_bytes(b"changed-checkpoint")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_checkpoint_hash_mismatch_not_released",
            )

            checkpoint.write_bytes(b"verified-checkpoint")
            invalid_metrics = {**report, "validation_loss": float("nan")}
            report_path.write_text(json.dumps(invalid_metrics), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint),
                "training_report_metrics_invalid_not_released",
            )

            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "code", 0, checkpoint_path=checkpoint, started_at_ns=time.time_ns() + 1_000_000),
                "training_report_stale_not_released",
            )

            self.assertEqual(
                _training_outcome(output, "code", 1, checkpoint_path=checkpoint),
                "training_failed_not_released",
            )

    def test_training_outcome_rejects_missing_held_out_metric_for_chat(self):
        import hashlib
        import json
        from aethrion.train_family import _training_outcome

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / "checkpoint_last.pt"
            checkpoint.write_bytes(b"chat-checkpoint")
            report = {
                "status": "training_complete_not_released",
                "release_status": "NOT_RELEASED",
                "model_id": "chat",
                "step": 9,
                "dataset_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "metrics": {"train": 1.2, "val": 1.1},
            }
            (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _training_outcome(output, "chat", 0, checkpoint_path=checkpoint),
                "training_report_metrics_invalid_not_released",
            )


    def test_interrupted_report_still_requires_matching_checkpoint_hash(self):
        from aethrion.train_family import _training_outcome

        with tempfile.TemporaryDirectory() as folder:
            output, checkpoint, _ = self._write_verified_report_fixture(
                Path(folder), status="training_interrupted_not_released"
            )
            self.assertEqual(
                _training_outcome(
                    output, "code", -2, checkpoint_path=checkpoint,
                    allow_interrupted=True,
                ),
                "training_interrupted_not_released",
            )
            checkpoint.write_bytes(b"tampered-after-report")
            self.assertEqual(
                _training_outcome(
                    output, "code", -2, checkpoint_path=checkpoint,
                    allow_interrupted=True,
                ),
                "training_checkpoint_hash_mismatch_not_released",
            )


if __name__ == "__main__":
    unittest.main()
