"""Strict numeric training configuration validation tests."""
import math
import unittest

from aethrion.training_config import (
    require_finite_number,
    require_int,
    validate_numeric_tree,
)


class StrictTrainingConfigValidationTests(unittest.TestCase):
    def test_require_int_accepts_only_exact_bounded_integers(self):
        self.assertEqual(require_int(7, "steps", minimum=1, maximum=10), 7)
        for value in (True, False, 7.0, "7", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    require_int(value, "steps", minimum=1, maximum=10)

    def test_require_int_rejects_values_outside_bounds(self):
        for value in (0, 11, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    require_int(value, "steps", minimum=1, maximum=10)

    def test_require_finite_number_rejects_non_numeric_and_non_finite_values(self):
        for value in (True, False, "0.1", None, math.nan, math.inf, -math.inf, 10**1000):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    require_finite_number(value, "learning_rate")

    def test_require_finite_number_enforces_inclusive_and_exclusive_bounds(self):
        self.assertEqual(
            require_finite_number(0.0, "weight_decay", minimum=0.0),
            0.0,
        )
        self.assertEqual(
            require_finite_number(1.0, "dropout", minimum=0.0, maximum=1.0),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            require_finite_number(0.0, "learning_rate", minimum=0.0, minimum_inclusive=False)
        with self.assertRaisesRegex(ValueError, "must be <= 1.0"):
            require_finite_number(1.01, "dropout", maximum=1.0)

    def test_validate_numeric_tree_accepts_nested_finite_json_numbers(self):
        validate_numeric_tree({
            "width": 128,
            "dropout": 0.1,
            "layers": [1, 2, 3],
            "nested": {"scale": 0.0},
        })

    def test_validate_numeric_tree_rejects_empty_objects_invalid_keys_and_non_numbers(self):
        invalid_values = (
            {},
            {"": 1},
            {1: 1},
            {"enabled": True},
            {"value": None},
            {"value": "128"},
            {"value": math.nan},
            {"value": math.inf},
            {"nested": {}},
        )
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    validate_numeric_tree(value)


if __name__ == "__main__":
    unittest.main()
