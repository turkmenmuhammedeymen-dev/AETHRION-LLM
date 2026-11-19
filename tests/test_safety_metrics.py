import unittest

import numpy as np

from aethrion.safety_metrics import SafetyEvaluationError, evaluate_safety_predictions


class SafetyMetricsTests(unittest.TestCase):
    def _balanced_fixture(self):
        labels = np.repeat(np.arange(3, dtype=np.int64), 25)
        probabilities = np.zeros((75, 3), dtype=np.float64)
        probabilities[np.arange(75), labels] = 0.9
        probabilities += 0.05
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return labels, probabilities

    def test_perfect_predictions_produce_zero_block_false_negatives(self):
        labels, probabilities = self._balanced_fixture()
        report = evaluate_safety_predictions(labels, probabilities)
        self.assertEqual(report["status"], "metrics_only_not_release_evidence")
        self.assertEqual(report["block_false_negative_rate"], 0.0)
        self.assertEqual(report["per_class"]["BLOCK"]["recall"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["confusion_matrix_true_rows_predicted_columns"], [
            [25, 0, 0], [0, 25, 0], [0, 0, 25]
        ])
        self.assertFalse(report["release_approval"])

    def test_low_confidence_predictions_abstain_to_review(self):
        labels, probabilities = self._balanced_fixture()
        probabilities[0] = [0.40, 0.35, 0.25]
        report = evaluate_safety_predictions(labels, probabilities, review_threshold=0.65)
        self.assertEqual(report["confusion_matrix_true_rows_predicted_columns"][0][1], 1)
        self.assertEqual(report["review_rate"], 1 / 75)

    def test_insufficient_per_class_support_is_rejected(self):
        labels = np.asarray([0] * 25 + [1] * 25 + [2] * 24, dtype=np.int64)
        probabilities = np.full((74, 3), 1 / 3, dtype=np.float64)
        with self.assertRaisesRegex(SafetyEvaluationError, "at least 25"):
            evaluate_safety_predictions(labels, probabilities)

    def test_invalid_probability_rows_are_rejected(self):
        labels, probabilities = self._balanced_fixture()
        probabilities[0] = [0.8, 0.8, 0.1]
        with self.assertRaisesRegex(SafetyEvaluationError, "sum to 1"):
            evaluate_safety_predictions(labels, probabilities)

    def test_nan_and_invalid_labels_are_rejected(self):
        labels, probabilities = self._balanced_fixture()
        probabilities[0, 0] = np.nan
        with self.assertRaisesRegex(SafetyEvaluationError, "finite"):
            evaluate_safety_predictions(labels, probabilities)
        labels[0] = 3
        with self.assertRaisesRegex(SafetyEvaluationError, "ALLOW=0"):
            evaluate_safety_predictions(labels, np.nan_to_num(probabilities))


if __name__ == "__main__":
    unittest.main()
