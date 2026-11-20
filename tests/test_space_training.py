import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aethrion.space_training import render_training_readiness, resolve_repository_root


class SpaceTrainingReadinessTests(unittest.TestCase):
    def test_repository_root_is_resolved_from_nested_space_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "aethrion").mkdir()
            (root / "aethrion" / "train_family.py").write_text("", encoding="utf-8")
            (root / "configs").mkdir()
            nested = root / "hf_space"
            nested.mkdir()
            app = nested / "app.py"
            app.write_text("", encoding="utf-8")
            self.assertEqual(resolve_repository_root(app), root.resolve())

    def test_readiness_calls_family_preflight_without_execution(self):
        fake = {
            "model_count": 11,
            "ready_count": 1,
            "blocked_count": 10,
            "models": [
                {"model_id": "chat", "ready": True, "status": "ready_not_started"},
                {"model_id": "code", "ready": False, "status": "blocked",
                 "reason": "FamilyTrainingError: source approval missing"},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "aethrion").mkdir()
            (root / "aethrion" / "train_family.py").write_text("", encoding="utf-8")
            (root / "configs").mkdir()
            app = root / "app.py"
            app.write_text("", encoding="utf-8")
            with patch("aethrion.train_family.run_family", return_value=fake) as run:
                rendered = render_training_readiness(app)
            run.assert_called_once_with(root.resolve(), execute=False)
            self.assertIn("Training started:** NO", rendered)
            self.assertIn("Checkpoints written:** NO", rendered)
            self.assertIn("chat | READY", rendered)
            self.assertIn("code | BLOCKED", rendered)
            self.assertIn("source approval missing", rendered)


if __name__ == "__main__":
    unittest.main()
