import tempfile
import unittest
from pathlib import Path

from aethrion.near_duplicate import find_cross_group_near_duplicates
from aethrion.prepare_data import build_corpus


def _long_record(changed_at=None):
    tokens = [f"token{index}" for index in range(100)]
    if changed_at is not None:
        tokens[changed_at] = "replacementtoken"
    return " ".join(tokens)


class NearDuplicateAuditTests(unittest.TestCase):
    def test_detects_near_duplicate_across_groups_deterministically(self):
        records = [
            {"text": _long_record(), "group_id": "author-a"},
            {"text": _long_record(50), "group_id": "author-b"},
            {"text": _long_record(20), "group_id": "author-a"},
        ]
        first = find_cross_group_near_duplicates(records)
        second = find_cross_group_near_duplicates(records)
        self.assertEqual(first, second)
        self.assertEqual([(item["left_index"], item["right_index"]) for item in first], [(0, 1)])
        self.assertGreaterEqual(first[0]["similarity"], 0.85)

    def test_ignores_short_generic_snippets(self):
        records = [
            {"text": "please read the terms and conditions carefully", "group_id": "a"},
            {"text": "please read the terms and conditions carefully today", "group_id": "b"},
        ]
        self.assertEqual(find_cross_group_near_duplicates(records), [])

    def test_rejects_invalid_similarity_threshold(self):
        with self.assertRaises(ValueError):
            find_cross_group_near_duplicates([], minimum_similarity=0)

    def _write_inputs(self, root, first_text, second_text):
        sources = []
        for index, value in enumerate((first_text, second_text, " ".join(f"separate{n}" for n in range(100)))):
            path = root / f"source-{index}.txt"
            path.write_text(value + "\n", encoding="utf-8")
            sources.append({
                "source_id": f"source-{index}",
                "path": path.name,
                "license": "Synthetic unit-test fixture",
                "permitted_uses": "training permitted for unit-test fixture only",
                "allow_training": False,
            })
        manifest = root / "sources.json"
        import json
        manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
        dataset_manifest = root / "dataset.json"
        dataset_manifest.write_text(json.dumps({"training_allowed": False}), encoding="utf-8")
        return manifest, dataset_manifest

    def test_build_corpus_fails_closed_on_cross_group_near_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, dataset_manifest = self._write_inputs(root, _long_record(), _long_record(50))
            with self.assertRaisesRegex(ValueError, "near-duplicate text crosses"):
                build_corpus(
                    root, manifest, root / "out/corpus.txt", root / "out/report.json",
                    min_chars=10, min_records=1, dataset_manifest_path=dataset_manifest,
                )

    def test_build_corpus_fails_closed_on_exact_duplicate_across_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, dataset_manifest = self._write_inputs(root, _long_record(), _long_record())
            with self.assertRaisesRegex(ValueError, "identical normalized content"):
                build_corpus(
                    root, manifest, root / "out/corpus.txt", root / "out/report.json",
                    min_chars=10, min_records=1, dataset_manifest_path=dataset_manifest,
                )


if __name__ == "__main__":
    unittest.main()
