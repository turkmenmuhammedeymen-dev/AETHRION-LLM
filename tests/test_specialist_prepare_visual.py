import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from aethrion.specialist_prepare_visual import VisualPreparationError, _resolve_asset, prepare_visual_dataset


class VisualPreparationTests(unittest.TestCase):
    def _kwargs(self, root):
        return {
            "output_root": root / "prepared",
            "license_name": "Unit test only",
            "permitted_use": "Temporary CI fixture only",
            "evidence_uri": "unit-test://visual-source",
            "source_uri": "unit-test://visual-fixture",
        }

    def _rows(self, root, model_id):
        rows = []
        for index in range(30):
            if model_id == "3d":
                points = np.random.default_rng(index).uniform(-1, 1, size=(64, 3)).astype(np.float32)
                np.save(root / f"points-{index}.npy", points, allow_pickle=False)
                rows.append({"id": f"object-{index}", "group_id": f"group-{index}", "point_path": f"points-{index}.npy",
                             "text": f"point cloud object {index}"})
            else:
                Image.new("RGB", (16, 12), (index, 20, 30)).save(root / f"image-{index}.png")
                rows.append({"id": f"image-{index}", "group_id": f"group-{index}", "image_path": f"image-{index}.png",
                             "caption": f"scene number {index}"})
        source = root / f"{model_id}.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return source

    def test_visual_asset_paths_reject_urls_windows_paths_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for value in ("../outside.png", "https://example.org/image.png", r"C:\images\sample.png", r"folder\sample.png"):
                with self.subTest(value=value):
                    with self.assertRaises(VisualPreparationError):
                        _resolve_asset(root, value, "sample")

    def test_visual_asset_symlink_cannot_escape_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inside"
            root.mkdir()
            outside = Path(directory) / "outside.png"
            outside.write_bytes(b"not an image")
            link = root / "linked.png"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks are unavailable on this runner.")
            with self.assertRaisesRegex(VisualPreparationError, "escapes"):
                _resolve_asset(root, "linked.png", "sample")

    def test_vision_and_image_assets_are_resized_and_normalized_to_uint8_nchw(self):
        for model_id, size in (("vision", 224), ("image", 64)):
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self._rows(root, model_id)
                result = prepare_visual_dataset(source, model_id=model_id, **self._kwargs(root))
                with np.load(result["dataset"], allow_pickle=False) as arrays:
                    self.assertEqual(arrays["images"].shape, (30, 3, size, size))
                    self.assertEqual(arrays["images"].dtype, np.uint8)
                    self.assertEqual(arrays["token_ids"].shape, (30, 128))
                manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
                self.assertFalse(manifest["training_allowed"])
                self.assertTrue(all(not item["allow_training"] for item in manifest["sources"]))

    def test_3d_point_clouds_are_resampled_and_stay_within_normalized_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._rows(root, "3d")
            result = prepare_visual_dataset(source, model_id="3d", **self._kwargs(root))
            with np.load(result["dataset"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["points"].shape, (30, 512, 3))
                self.assertTrue(np.isfinite(arrays["points"]).all())
                self.assertLessEqual(float(np.max(np.abs(arrays["points"]))), 1.0)
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["owner_approval"])


if __name__ == "__main__":
    unittest.main()
