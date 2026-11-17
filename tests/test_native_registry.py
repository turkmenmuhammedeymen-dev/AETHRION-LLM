import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aethrion.native_registry import (
    NativeModelError,
    NativeModelIntegrityError,
    NativeModelRegistry,
    NativeModelUnavailable,
)


def write_manifest(root: Path, entry: dict, *, capability: str = "chat") -> Path:
    models = {} if entry is None else {capability: entry}
    path = root / "models" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "format": "aethrion-native-model-family-v1",
        "execution_policy": "local_only",
        "models": models,
    }), encoding="utf-8")
    return path


class NativeModelRegistryTests(unittest.TestCase):
    def test_untrained_model_is_not_resolvable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "models/chat.pt",
                "status": "not_trained",
            })
            registry = NativeModelRegistry(root)
            self.assertEqual(registry.status("chat"), "not_trained")
            with self.assertRaises(NativeModelUnavailable):
                registry.resolve("chat")

    def test_valid_local_artifacts_resolve_after_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "chat.pt"
            tokenizer = root / "models" / "tokenizer.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"first-party-checkpoint")
            tokenizer.write_text('{"format":"aethrion-test"}', encoding="utf-8")
            entry = {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "models/chat.pt",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "tokenizer": "models/tokenizer.json",
                "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                "status": "trained",
            }
            write_manifest(root, entry)
            registry = NativeModelRegistry(root)
            resolved = registry.resolve("chat")
            self.assertEqual(resolved.checkpoint, checkpoint.resolve())
            self.assertEqual(resolved.tokenizer, tokenizer.resolve())
            self.assertEqual(registry.status("chat"), "ready")

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "chat.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"tampered")
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "models/chat.pt",
                "checkpoint_sha256": "0" * 64,
                "status": "trained",
            })
            registry = NativeModelRegistry(root)
            self.assertEqual(registry.status("chat"), "integrity_failed")
            with self.assertRaises(NativeModelIntegrityError):
                registry.resolve("chat")

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "../outside.pt",
                "checkpoint_sha256": "0" * 64,
                "status": "trained",
            })
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)

    def test_remote_url_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, {
                "architecture": "aethrion.model:AETHRIONLM",
                "checkpoint": "https://example.invalid/model.pt",
                "checkpoint_sha256": "0" * 64,
                "status": "trained",
            })
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)

    def test_unknown_capability_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "models" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "format": "aethrion-native-model-family-v1",
                "execution_policy": "local_only",
                "models": {"unknown": {}},
            }), encoding="utf-8")
            with self.assertRaises(NativeModelError):
                NativeModelRegistry(root)


if __name__ == "__main__":
    unittest.main()
