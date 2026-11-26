import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BinaryAssetAuditTests(unittest.TestCase):
    def test_full_audit_reports_no_issues(self):
        try:
            import PIL  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("Pillow/numpy not installed")
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/audit_binary_assets.py")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:])
        self.assertIn("ISSUES: 0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
