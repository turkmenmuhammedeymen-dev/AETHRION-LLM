import csv
import json
import tempfile
import unittest
from pathlib import Path

from aethrion.prepare_data import (
    build_corpus,
    load_sources_manifest,
    normalize_text,
    require_training_enabled,
)


class CorpusPreparationTests(unittest.TestCase):
    def _write_dataset_manifest(self, root: Path, allowed: bool = True) -> Path:
        path = root / "corpus_manifest.json"
        path.write_text(
            json.dumps({
                "project": "AETHRION-LLM-tests",
                "training_allowed": allowed,
            }),
            encoding="utf-8",
        )
        return path

    def test_unicode_and_whitespace(self):
        self.assertEqual(normalize_text("  Türkçe\n  metin  "), "Türkçe metin")

    def test_manifest_pipeline_deduplicates_and_filters_email(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text(
                ("Bu, AETHRION için yeterince uzun bir Türkçe eğitim cümlesidir.\n") * 2
                + "İkinci ve farklı bir Türkçe eğitim örneği burada bulunuyor.\n"
                + "İletişim test@example.com olan kayıt dışarıda kalmalıdır.\n",
                encoding="utf-8",
            )
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [{
                "source_id": "unit-test",
                "path": "source.txt",
                "license": "test",
                "permitted_uses": "unit test",
                "allow_training": True,
            }]}), encoding="utf-8")
            output = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            report = build_corpus(
                root, manifest, output, report_path, seed=7, min_chars=10,
                min_records=1, dataset_manifest_path=self._write_dataset_manifest(root),
            )
            self.assertEqual(report["totals"]["records_kept"], 2)
            self.assertEqual(report["totals"]["duplicates"], 1)
            self.assertEqual(report["totals"]["email_like_records_excluded"], 1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            self.assertTrue(report_path.exists())

    def test_rejects_unapproved_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text("some text", encoding="utf-8")
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [{
                "source_id": "blocked",
                "path": "source.txt",
                "license": "unknown",
                "permitted_uses": "none",
                "allow_training": False,
            }]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sources_manifest(manifest, root)

    def _write_url_source(self, root: Path):
        source = root / "mixed.jsonl"
        source.write_text(json.dumps({
            "url": "https://example.org/article",
            "text": "Bu kayıt kaynak alan adı onayı olmadan eğitime giremez.",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = root / "sources.json"
        manifest.write_text(json.dumps({"sources": [{
            "source_id": "mixed-test",
            "path": "mixed.jsonl",
            "license": "dataset-label",
            "permitted_uses": "review required",
            "allow_training": True,
        }]}), encoding="utf-8")
        return source, manifest

    def test_url_records_require_domain_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_url_source(root)
            with self.assertRaisesRegex(ValueError, "--source-review"):
                build_corpus(
                    root, root / "sources.json", root / "out/corpus.txt",
                    root / "out/report.json", min_chars=10, min_records=1,
                    dataset_manifest_path=self._write_dataset_manifest(root),
                )

    def test_pending_domain_review_blocks_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_url_source(root)
            review = root / "review.csv"
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "domain", "license_name", "license_evidence_url", "rights_holder",
                    "ai_training_permission", "attribution_required", "review_status",
                ])
                writer.writeheader()
                writer.writerow({
                    "domain": "example.org",
                    "license_name": "UNKNOWN",
                    "license_evidence_url": "",
                    "rights_holder": "UNKNOWN",
                    "ai_training_permission": "NOT_VERIFIED",
                    "attribution_required": "UNKNOWN",
                    "review_status": "PENDING",
                })
            with self.assertRaisesRegex(ValueError, "Training blocked"):
                build_corpus(
                    root, root / "sources.json", root / "out/corpus.txt",
                    root / "out/report.json", min_chars=10, min_records=1,
                    source_review_path=review,
                    dataset_manifest_path=self._write_dataset_manifest(root),
                )

    def test_evidence_backed_domain_approval_allows_reviewed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_url_source(root)
            review = root / "review.csv"
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "domain", "license_name", "license_evidence_url", "rights_holder",
                    "ai_training_permission", "attribution_required", "review_status",
                ])
                writer.writeheader()
                writer.writerow({
                    "domain": "example.org",
                    "license_name": "CC0",
                    "license_evidence_url": "https://example.org/license",
                    "rights_holder": "Example rights holder",
                    "ai_training_permission": "YES",
                    "attribution_required": "NO",
                    "review_status": "APPROVED",
                })
            report = build_corpus(
                root, root / "sources.json", root / "out/corpus.txt",
                root / "out/report.json", min_chars=10, min_records=1,
                source_review_path=review,
                dataset_manifest_path=self._write_dataset_manifest(root),
            )
            self.assertEqual(report["domain_review"][0]["domain"], "example.org")
            self.assertEqual(report["domain_review"][0]["records"], 1)


    def test_dataset_manifest_requires_exact_boolean_true(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing = root / "missing.json"

            with self.assertRaises(ValueError):
                require_training_enabled(missing)

            invalid_values = [
                {},
                {"training_allowed": False},
                {"training_allowed": "true"},
                {"training_allowed": 1},
                {"training_allowed": None},
            ]

            for index, payload in enumerate(invalid_values):
                path = root / f"invalid_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        require_training_enabled(path)

    def test_dataset_manifest_accepts_exact_boolean_true(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "approved.json"
            payload = {
                "project": "AETHRION-LLM-tests",
                "training_allowed": True,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = require_training_enabled(path)
            self.assertIs(result["training_allowed"], True)


if __name__ == "__main__":
    unittest.main()
