import tempfile
import unittest
from pathlib import Path

from aethrion.visual_safety import (
    VisualSafetyError, inspect_visual_artifact, require_visual_output_allow,
)


class _ReleasedVisualVerifier:
    def __init__(self, verdict=None):
        self.verdict = verdict or {"decision": "ALLOW", "confidence": 0.99}
    def visual_safety_status(self):
        return "ready"
    def review_visual_artifact(self, path, modality, inspection):
        self.asserted = (path, modality, inspection)
        return self.verdict


class VisualSafetyTests(unittest.TestCase):
    def test_missing_semantic_verifier_fails_closed_after_structural_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloud.obj"
            path.write_text("v 0 0 0\nv 1 1 1\n", encoding="utf-8")
            with self.assertRaisesRegex(VisualSafetyError, "not trained and released"):
                require_visual_output_allow(object(), path, "3d")

    def test_3d_point_cloud_accepts_finite_vertices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloud.obj"
            path.write_text("v 0 0 0\nv 1.5 -2 3\n", encoding="utf-8")
            result = inspect_visual_artifact(path, "3d")
            self.assertEqual(result["vertices"], 2)

    def test_3d_rejects_non_finite_coordinates_and_mesh_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloud.obj"
            path.write_text("v nan 0 0\n", encoding="utf-8")
            with self.assertRaises(VisualSafetyError):
                inspect_visual_artifact(path, "3d")
            path.write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
            with self.assertRaisesRegex(VisualSafetyError, "vertex-only"):
                inspect_visual_artifact(path, "3d")

    def test_3d_semantic_verifier_must_allow_with_high_confidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloud.obj"
            path.write_text("v 0 0 0\n", encoding="utf-8")
            verifier = _ReleasedVisualVerifier()
            result = require_visual_output_allow(verifier, path, "3d")
            self.assertEqual(result["verdict"]["decision"], "ALLOW")
            verifier.verdict = {"decision": "ALLOW", "confidence": 0.5}
            with self.assertRaises(VisualSafetyError):
                require_visual_output_allow(verifier, path, "3d")
            verifier.verdict = {"decision": "BLOCK", "confidence": 0.99}
            with self.assertRaises(VisualSafetyError):
                require_visual_output_allow(verifier, path, "3d")

    def test_image_rejects_unexpected_format_or_corrupt_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(VisualSafetyError):
                inspect_visual_artifact(path, "image")


if __name__ == "__main__":
    unittest.main()
