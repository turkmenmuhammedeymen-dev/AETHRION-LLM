import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aethrion.space_artifacts import (
    ArtifactStoreError,
    _configuration,
    _model_id,
    _private_dataset_api,
    save_checkpoint,
    restore_checkpoint,
)


class _PrivateInfo:
    private = True


class _PublicInfo:
    private = False


class _FakeApi:
    def __init__(self, token):
        self.files = {}
        self.token = token

    def repo_info(self, repo_id, repo_type):
        return _PrivateInfo()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()


class SpaceArtifactTests(unittest.TestCase):
    def test_configuration_requires_secret_and_repo(self):
        with self.assertRaises(ArtifactStoreError):
            _configuration({})
        self.assertEqual(
            _configuration({"HF_TOKEN": "secret", "AETHRION_CHECKPOINT_REPO": "owner/private"}),
            ("secret", "owner/private"),
        )

    def test_model_allowlist(self):
        self.assertEqual(_model_id("chat"), "chat")
        for value in ("../chat", "unknown", "CHAT", ""):
            with self.subTest(value=value), self.assertRaises(ArtifactStoreError):
                _model_id(value)

    def test_public_repository_is_rejected(self):
        class Api:
            def __init__(self, token): pass
            def repo_info(self, repo_id, repo_type): return _PublicInfo()
        with patch("huggingface_hub.HfApi", Api):
            with self.assertRaises(ArtifactStoreError):
                _private_dataset_api("secret", "owner/public")

    def test_save_uploads_pointer_last_and_restore_checks_hash(self):
        api = _FakeApi("secret")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"first-party-test-checkpoint")
            env = {"HF_TOKEN": "secret", "AETHRION_CHECKPOINT_REPO": "owner/private"}
            with patch("huggingface_hub.HfApi", lambda token: api):
                saved = save_checkpoint("chat", checkpoint, environ=env)
            self.assertEqual(saved["status"], "stored_private")
            pointer_path = "checkpoints/chat/latest.json"
            pointer = json.loads(api.files[pointer_path])
            self.assertEqual(pointer["sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            restored = root / "restored.pt"
            with patch("huggingface_hub.HfApi", lambda token: api), patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo_id, repo_type, filename, token: self._materialize(api, filename, root),
            ):
                result = restore_checkpoint("chat", restored, environ=env)
            self.assertEqual(result["status"], "restored_verified")
            self.assertEqual(restored.read_bytes(), checkpoint.read_bytes())

    @staticmethod
    def _materialize(api, filename, root):
        target = root / ("remote-" + filename.replace("/", "_"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(api.files[filename])
        return str(target)


if __name__ == "__main__":
    unittest.main()
