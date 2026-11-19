import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SyntheticBasePilotTests(unittest.TestCase):
    def test_synthetic_base_pilot_is_labeled_and_unapproved(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "synthetic"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_synthetic_base_pilot.py"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "sources.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["dataset_type"], "synthetic_only_not_owner_authored"
            )
            self.assertIs(manifest["training_allowed"], False)
            self.assertEqual(len(manifest["sources"]), 3)

            total = 0
            for source in manifest["sources"]:
                self.assertEqual(source["origin"], "ai_generated_synthetic")
                self.assertIs(source["allow_training"], False)
                rows = (
                    output / source["path"]
                ).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), source["record_count"])
                self.assertTrue(all(len(row) >= 20 for row in rows))
                total += len(rows)

            self.assertGreaterEqual(total, 100)


if __name__ == "__main__":
    unittest.main()
