import tempfile
import unittest
from pathlib import Path

from ci.trigger_space_training import _manifest_artifact_path


class ManifestArtifactPathTests(unittest.TestCase):
    def test_manifest_paths_resolve_from_shared_manifests_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "data" / "specialists" / "manifests" / "code" / "source.jsonl"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"id":"code-0001"}\\n', encoding="utf-8")
            resolved = _manifest_artifact_path(root, "code/source.jsonl")
            self.assertEqual(resolved, artifact.resolve())
            self.assertTrue(resolved.is_file())

    def test_manifest_paths_cannot_escape_manifests_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "escapes"):
                _manifest_artifact_path(root, "../outside.jsonl")


if __name__ == "__main__":
    unittest.main()
