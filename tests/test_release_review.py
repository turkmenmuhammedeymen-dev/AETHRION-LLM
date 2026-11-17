import json
from pathlib import Path
import tempfile
import unittest

from aethrion.native_registry import REQUIRED_BENCHMARKS, REQUIRED_BENCHMARK_DIRECTIONS
from aethrion.release_review import ReleaseReviewError, create_release_evaluation_template


class ReleaseReviewTemplateTests(unittest.TestCase):
    def test_creates_pending_review_with_model_specific_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.json"
            training.write_text(json.dumps({
                "status": "training_complete_not_released",
                "model_id": "asr",
                "dataset_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
            }), encoding="utf-8")
            output = root / "evaluation.json"
            report = create_release_evaluation_template("asr", training, output)
            self.assertEqual(report["status"], "pending_review")
            self.assertFalse(report["quality_gate_passed"])
            self.assertFalse(report["safety_gate_passed"])
            self.assertFalse(report["human_reviewed"])
            self.assertEqual(set(report["benchmarks"]), {"wer", "cer"})
            self.assertTrue(output.is_file())

    def test_every_required_benchmark_has_a_valid_direction(self):
        self.assertEqual(set(REQUIRED_BENCHMARK_DIRECTIONS), set(REQUIRED_BENCHMARKS))
        for model_id, required in REQUIRED_BENCHMARKS.items():
            directions = REQUIRED_BENCHMARK_DIRECTIONS[model_id]
            self.assertEqual(set(directions), required)
            self.assertTrue(set(directions.values()).issubset({"min", "max"}))
        self.assertEqual(REQUIRED_BENCHMARK_DIRECTIONS["code"]["security_regression"], "min")

    def test_refuses_in_progress_training_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.json"
            training.write_text(json.dumps({
                "status": "training_in_progress_not_released",
                "model_id": "code",
                "dataset_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
            }), encoding="utf-8")
            with self.assertRaises(ReleaseReviewError):
                create_release_evaluation_template("code", training, root / "eval.json")


if __name__ == "__main__":
    unittest.main()
