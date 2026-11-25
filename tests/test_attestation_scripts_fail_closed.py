import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class AttestationFailClosedTests(unittest.TestCase):
    def run_script(self, path, *args):
        env = dict(os.environ)
        env.pop("AETHRION_BASE_AUDIT_HMAC_KEY", None)
        return subprocess.run([sys.executable, str(ROOT / path), *args], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_owner_attestation_command_cannot_change_files_or_approval_flags(self):
        governance = ROOT / "data/authored/chat/GOVERNANCE.json"
        manifest = ROOT / "data/specialists/manifests/chat.json"
        before = (hashlib.sha256(governance.read_bytes()).hexdigest(),
                  hashlib.sha256(manifest.read_bytes()).hexdigest())
        result = self.run_script("scripts/apply_owner_attestation.py",
                                 "chat", "--session", "test-session", "--statement", "I approve")
        after = (hashlib.sha256(governance.read_bytes()).hexdigest(),
                 hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Automatic dataset attestation is disabled", result.stderr)
        self.assertEqual(before, after)

    def test_base_preparer_never_opens_training_gate(self):
        manifest = ROOT / "data/processed/corpus_manifest.json"
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result = self.run_script("scripts/prepare_base.py")
        after = hashlib.sha256(manifest.read_bytes()).hexdigest()
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
