import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from aethrion import space_session as ss, artifact_sync as s
from tests.test_artifact_sync import Fake, ENV

class SessionTests(unittest.TestCase):
    def _run_session(self, *args, **kwargs):
        # The fake Hub client requires the same explicit private-repo setting
        # that production receives from the Space environment.
        kwargs.setdefault("environ", ENV)
        return ss.run_session(*args, **kwargs)

    def _config(self, root, model, output_dir, *, config_model=None):
        path = (Path(root) / "configs" / ("small.json" if model == "chat" else f"specialists/{model}.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model_id": config_model or model, "output_dir": output_dir}),
            encoding="utf-8",
        )

    def test_gate_failure_pushes_nothing_and_never_releases(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "chat", "artifacts")
            r = self._run_session("chat", 10, root=d, api=Fake(), environ=ENV,
                               runner=lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr="gate closed"))
            self.assertFalse(r["trainer_ok"]); self.assertEqual(r["pushed"], []); self.assertFalse(r["released"])
    def test_success_pushes_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "chat", "artifacts")
            def run(cmd, **k):
                p = Path(d, "artifacts"); p.mkdir(parents=True, exist_ok=True); (p / "checkpoint_last.pt").write_bytes(b"x")
                return SimpleNamespace(returncode=0, stdout="done", stderr="")
            r = self._run_session("chat", 10, root=d, api=Fake(), environ=ENV, runner=run)
            self.assertEqual(r["pushed"], ["checkpoint_last.pt"]); self.assertFalse(r["released"])
    def test_specialist_uses_its_configured_output_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "code", "artifacts/models/code")
            def run(cmd, **k):
                p = Path(d, "artifacts", "models", "code")
                p.mkdir(parents=True, exist_ok=True)
                (p / "checkpoint_last.pt").write_bytes(b"code-checkpoint")
                return SimpleNamespace(returncode=0, stdout="done", stderr="")
            r = self._run_session("code", 5, root=d, api=Fake(), environ=ENV, runner=run)
            self.assertEqual(r["pushed"], ["checkpoint_last.pt"])

    def test_invalid_or_missing_config_fails_before_sync(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("aethrion.space_session.artifact_sync.pull_model_artifacts") as pull:
                with self.assertRaises(s.ArtifactSyncError):
                    self._run_session("chat", 5, root=d, api=Fake(), environ=ENV, runner=lambda *a, **k: None)
                pull.assert_not_called()

    def test_legacy_base_identifier_is_rejected_before_artifact_download(self):
        with patch("aethrion.space_session.artifact_sync.pull_model_artifacts") as pull:
            with self.assertRaisesRegex(s.ArtifactSyncError, "unsupported training model"):
                self._run_session("base", 5, api=Fake(), environ=ENV)
            pull.assert_not_called()

    def test_specialist_config_model_mismatch_is_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as d:
            # The selected model determines the config filename; its contents
            # deliberately claim a different model to exercise the mismatch guard.
            self._config(d, "code", "artifacts/models/code", config_model="vision")
            with patch("aethrion.space_session.artifact_sync.pull_model_artifacts") as pull:
                with self.assertRaisesRegex(s.ArtifactSyncError, "does not match"):
                    self._run_session("code", 5, root=d, api=Fake(), environ=ENV)
                pull.assert_not_called()

    def test_validation(self):
        for m, n in (("evil", 5), ("chat", 0), ("chat", 99999), ("chat", True)):
            with self.assertRaises((s.ArtifactSyncError, ValueError)):
                self._run_session(m, n, api=Fake(), environ=ENV)

if __name__ == "__main__": unittest.main()
