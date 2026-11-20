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

    def test_explicit_empty_selection_is_rejected_instead_of_running_all_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("aethrion.train_family.subprocess.run") as run:
                with self.assertRaisesRegex(FamilyTrainingError, "empty model selection"):
                    run_family(tmp, selected=[], execute=True)
            run.assert_not_called()

    def test_malformed_selection_and_execution_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for selected in ("chat", [None], [1]):
                with self.subTest(selected=selected):
                    with self.assertRaises(FamilyTrainingError):
                        run_family(tmp, selected=selected)
            with self.assertRaisesRegex(FamilyTrainingError, "execute"):
                run_family(tmp, execute=1)
            with self.assertRaisesRegex(FamilyTrainingError, "python_executable"):
                run_family(tmp, python_executable="")

    def test_output_directory_must_be_explicit_and_non_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for invalid in ("", ".", "..", "../outside", str(root / "absolute")):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(FamilyTrainingError):
                        _resolve_output_dir(root, invalid)

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


if __name__ == "__main__":
    unittest.main()
