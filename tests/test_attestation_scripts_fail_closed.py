import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def file_state(path: Path):
    """Return a content hash, or None when a deliberately absent artifact stays absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class AttestationFailClosedTests(unittest.TestCase):
    def run_script(self, path, *args):
        env = dict(os.environ)
        env.pop("AETHRION_BASE_AUDIT_HMAC_KEY", None)
        return subprocess.run([sys.executable, str(ROOT / path), *args], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_owner_attestation_command_cannot_change_files_or_approval_flags(self):
        governance = ROOT / "data/authored/chat/GOVERNANCE.json"
        manifest = ROOT / "data/specialists/manifests/chat.json"
        before = (file_state(governance), file_state(manifest))
        result = self.run_script("scripts/apply_owner_attestation.py",
                                 "chat", "--session", "test-session", "--statement", "I approve")
        after = (file_state(governance), file_state(manifest))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Automatic dataset attestation is disabled", result.stderr)
        self.assertEqual(before, after)
        # CHAT preparation is intentionally blocked; an absent manifest must remain absent.
        self.assertEqual(manifest.exists(), before[1] is not None)

    def test_base_preparer_never_opens_training_gate(self):
        manifest = ROOT / "data/processed/corpus_manifest.json"
        before = file_state(manifest)
        result = self.run_script("scripts/prepare_base.py")
        after = file_state(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("automatic owner attestation has been disabled", result.stderr)
        self.assertEqual(before, after)

    def test_chat_ci_runner_does_not_invoke_automatic_attestation(self):
        source = (ROOT / "ci/instruction_tune_chat.py").read_text(encoding="utf-8")
        self.assertNotIn("scripts/apply_owner_attestation.py", source)
        self.assertIn("validate_governance_record", source)
        self.assertIn("CHAT human review is pending or incomplete", source)


if __name__ == "__main__":
    unittest.main()
