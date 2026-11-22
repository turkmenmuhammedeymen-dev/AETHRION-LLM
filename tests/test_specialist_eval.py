import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    from aethrion.specialist_eval import evaluate_specialist, _macro_f1
    from aethrion.specialist_train import train_specialist
    from tests.test_specialist_train import SpecialistTrainingTests
    HAVE_TORCH = True
except Exception:  # torch is validated in CI only
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class SpecialistEvalTests(unittest.TestCase):
    def test_evaluation_is_read_only_and_never_releases(self):
        helper = SpecialistTrainingTests()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = helper._fixture(root, True)
            config = helper._config(root)
            train_specialist(config, manifest)
            ckpt = Path(config["output_dir"]) / "checkpoint_last.pt"
            before = ckpt.read_bytes()
            report = evaluate_specialist(config, manifest, ckpt)
            self.assertEqual(report["status"], "evaluated_not_released")
            self.assertEqual(report["release_status"], "NOT_RELEASED")
            self.assertTrue(report["weights_finite"])
            self.assertEqual(report["test_samples"], 2)
            self.assertIn("task_metrics", report)
            self.assertEqual(ckpt.read_bytes(), before)

    def test_macro_f1_perfect_and_zero(self):
        import numpy as np
        self.assertEqual(_macro_f1(np.array([0, 1]), np.array([0, 1])), 1.0)
        self.assertEqual(_macro_f1(np.array([0, 1]), np.array([1, 0])), 0.0)
