import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from aethrion import space_session as ss, artifact_sync as s
from tests.test_artifact_sync import Fake, ENV

class SessionTests(unittest.TestCase):
    def _config(self, root, model, output_dir):
        path = (Path(root) / "configs" / ("small.json" if model == "chat" else f"specialists/{model}.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_id": model, "output_dir": output_dir}), encoding="utf-8")

    def test_gate_failure_pushes_nothing_and_never_releases(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "chat", "artifacts")
            r = ss.run_session("chat", 10, root=d, api=Fake(),
                               runner=lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr="gate closed"))
            self.assertFalse(r["trainer_ok"]); self.assertEqual(r["pushed"], []); self.assertFalse(r["released"])
    def test_success_pushes_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "chat", "artifacts")
            def run(cmd, **k):
                p = Path(d, "artifacts"); p.mkdir(parents=True, exist_ok=True); (p / "checkpoint_last.pt").write_bytes(b"x")
                return SimpleNamespace(returncode=0, stdout="done", stderr="")
            r = ss.run_session("chat", 10, root=d, api=Fake(), environ=ENV, runner=run)
            self.assertEqual(r["pushed"], ["checkpoint_last.pt"]); self.assertFalse(r["released"])
    def test_specialist_uses_its_configured_output_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self._config(d, "code", "artifacts/models/code")
            def run(cmd, **k):
                p = Path(d, "artifacts", "models", "code")
                p.mkdir(parents=True, exist_ok=True)
                (p / "checkpoint_last.pt").write_bytes(b"code-checkpoint")
                return SimpleNamespace(returncode=0, stdout="done", stderr="")
            r = ss.run_session("code", 5, root=d, api=Fake(), environ=ENV, runner=run)
            self.assertEqual(r["pushed"], ["checkpoint_last.pt"])

    def test_invalid_or_missing_config_fails_before_sync(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(s.ArtifactSyncError):
                ss.run_session("chat", 5, root=d, api=Fake(), environ=ENV, runner=lambda *a, **k: None)

    def test_validation(self):
        for m, n in (("evil", 5), ("chat", 0), ("chat", 99999), ("chat", True)):
            with self.assertRaises((s.ArtifactSyncError, ValueError)):
                ss.run_session(m, n, api=Fake(), environ=ENV)

if __name__ == "__main__": unittest.main()
