import unittest

from aethrion.space_auth import space_auth_required


class SpaceAuthenticationGateTests(unittest.TestCase):
    def test_missing_authentication_fails_closed(self):
        for value in (False, None, 0, "", "true"):
            with self.subTest(value=value):
                self.assertTrue(space_auth_required(value))

    def test_only_explicit_true_enables_access(self):
        self.assertFalse(space_auth_required(True))


    def test_diagnostics_workflow_uses_space_basic_auth_and_safe_restart_policy(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "diagnose-hf-space.yml").read_text(encoding="utf-8")
        self.assertIn("AETHRION_SPACE_USERNAME", workflow)
        self.assertIn("AETHRION_SPACE_PASSWORD", workflow)
        self.assertIn("auth=(space_username, space_password)", workflow)
        self.assertNotIn('headers = {"Authorization": f"Bearer {token}"}', workflow)
        self.assertIn('runtime.hardware == "cpu-basic"', workflow)
        self.assertIn("factory_reboot=False", workflow)
        self.assertIn("hardware_may_be_billable", workflow)
        self.assertIn("redact(", workflow)
        self.assertIn("concurrency:", workflow)


if __name__ == "__main__":
    unittest.main()
