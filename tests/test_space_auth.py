import unittest

from aethrion.space_auth import space_auth_required


class SpaceAuthenticationGateTests(unittest.TestCase):
    def test_missing_authentication_fails_closed(self):
        for value in (False, None, 0, "", "true"):
            with self.subTest(value=value):
                self.assertTrue(space_auth_required(value))

    def test_only_explicit_true_enables_access(self):
        self.assertFalse(space_auth_required(True))


if __name__ == "__main__":
    unittest.main()
