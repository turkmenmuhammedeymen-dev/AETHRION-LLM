import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aethrion.train_family import MODEL_IDS, preflight_model, run_family


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


if __name__ == "__main__":
    unittest.main()
