import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "safety" / "policy_v1.json"


class SafetyPolicyConfigTests(unittest.TestCase):
    def test_policy_is_explicitly_draft_and_not_enforced(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(policy["format"], "aethrion-safe-policy-v1")
        self.assertEqual(policy["status"], "POLICY_DRAFT_NOT_A_TRAINED_OR_ENFORCED_MODEL")
        self.assertEqual(policy["decisions"], ["ALLOW", "REVIEW", "BLOCK"])

    def test_each_policy_category_has_safe_alternative_and_decision(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(policy["categories"]), 8)
        ids = set()
        for category in policy["categories"]:
            self.assertTrue(category["id"])
            self.assertNotIn(category["id"], ids)
            ids.add(category["id"])
            self.assertTrue(category["description"])
            self.assertIn(category["default_for_harmful_intent"], policy["decisions"])
            self.assertTrue(category["safe_alternative"])

    def test_release_requires_independent_evaluation_and_approval(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        gates = policy["release_gates"]
        self.assertTrue(any("held-out test" in item for item in gates))
        self.assertTrue(any("human review" in item.lower() for item in gates))
        self.assertTrue(any("release approval" in item.lower() for item in gates))


if __name__ == "__main__":
    unittest.main()
