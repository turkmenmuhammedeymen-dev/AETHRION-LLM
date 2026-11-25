import tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from aethrion import artifact_sync as s

ENV = {"AETHRION_ARTIFACT_REPO": "owner/private-artifacts"}

class Fake:
    def __init__(self, private=True): self.private = private; self.up = []; self.files = {}
    def create_repo(self, **k): self.created = k
    def repo_info(self, **k): return SimpleNamespace(private=self.private)
    def upload_file(self, **k): self.up.append(k["path_in_repo"]); self.files[k["path_in_repo"]] = Path(k["path_or_fileobj"]).read_bytes()
    def list_repo_files(self, **k): return list(self.files)
    def hf_hub_download(self, repo_id, repo_type, filename, local_dir):
        p = Path(local_dir) / filename.replace("/", "_"); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(self.files[filename]); return str(p)

class ArtifactSyncTests(unittest.TestCase):
    def test_round_trip_and_private_creation(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "src"); src.mkdir(); (src / "checkpoint_last.pt").write_bytes(b"w"); (src / "checkpoint_last.pt.sha256").write_text("h")
            (src / "notes.txt").write_text("must be rejected")
            (src / "metrics.jsonl").write_text('{"step":1,"train_loss":0.5}\\n')
            (src / "training_report.json").write_text('{"status":"training_in_progress_not_released"}')
            (src / "progress.json").write_text('{"step":1500}')
            f = Fake()
            with self.assertRaises(s.ArtifactSyncError): s.push_model_artifacts("chat", src, api=f, environ=ENV)
            (src / "notes.txt").unlink()
            self.assertEqual(sorted(s.push_model_artifacts("chat", src, api=f, environ=ENV)), ["checkpoint_last.pt", "checkpoint_last.pt.sha256", "metrics.jsonl", "progress.json", "training_report.json"])
            self.assertTrue(f.created["private"])
            dst = Path(d, "dst", "chat")
            self.assertEqual(sorted(s.pull_model_artifacts("chat", dst, api=f, environ=ENV)), ["checkpoint_last.pt", "checkpoint_last.pt.sha256", "metrics.jsonl", "progress.json", "training_report.json"])
            self.assertEqual((dst / "checkpoint_last.pt").read_bytes(), b"w")
    def test_refuses_public_repo_unknown_model_and_missing_config(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(s.ArtifactSyncError): s.push_model_artifacts("chat", d, api=Fake(private=False), environ=ENV)
            with self.assertRaises(s.ArtifactSyncError): s.push_model_artifacts("evil", d, api=Fake(), environ=ENV)
            with self.assertRaises(s.ArtifactSyncError): s.push_model_artifacts("chat", d, api=Fake(), environ={})
    def test_rejects_unknown_files_before_any_upload(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "checkpoint_last.pt").write_bytes(b"checkpoint")
            (root / "unexpected.json").write_text("{}")
            fake = Fake()
            with self.assertRaises(s.ArtifactSyncError):
                s.push_model_artifacts("chat", root, api=fake, environ=ENV)
            self.assertEqual(fake.up, [])

    def test_rejects_bad_names(self):
        for n in ("../x.pt", ".hidden.pt", "a b.pt", "x.exe"):
            with self.assertRaises(s.ArtifactSyncError): s._safe("chat", n)

if __name__ == "__main__": unittest.main()


class SingleCommitTests(unittest.TestCase):
    def test_push_uses_one_commit_for_all_files(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest('huggingface_hub is validated in CI')
        import tempfile
        s = __import__("aethrion.artifact_sync", fromlist=["x"])
        class C(Fake):
            commits = []
            def create_commit(self, **k): self.commits.append(k)
        with tempfile.TemporaryDirectory() as d:
            for name in ("checkpoint_last.pt", "checkpoint_last.pt.sha256", "tokenizer.json", "tokenizer.json.provenance.json"):
                (Path(d) / name).write_bytes(b"x")
            c = C()
            sent = s.push_model_artifacts("code", d, api=c, environ=ENV)
        self.assertEqual(len(C.commits), 1)
        self.assertEqual(sorted(sent), ["checkpoint_last.pt", "checkpoint_last.pt.sha256", "tokenizer.json", "tokenizer.json.provenance.json"])
        self.assertEqual(c.up, [])


class TokenizerPackageSyncTests(unittest.TestCase):
    def test_chat_push_includes_tokenizer_provenance(self):
        import tempfile
        s = __import__("aethrion.artifact_sync", fromlist=["x"])
        with tempfile.TemporaryDirectory() as d:
            names = ("checkpoint_last.pt", "checkpoint_last.pt.sha256", "tokenizer.json", "tokenizer.json.provenance.json")
            for name in names:
                (Path(d) / name).write_bytes(b"x")
            f = Fake()
            sent = s.push_model_artifacts("chat", d, api=f, environ=ENV)
        self.assertEqual(sorted(sent), sorted(names))
