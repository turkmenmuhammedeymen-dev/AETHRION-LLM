from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aethrion import pipeline


class PipelineApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "corpus_manifest.json"
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.sources = self.raw / "sources.json"
        self.review = self.root / "source_review.csv"

        self.patches = [
            patch.object(pipeline, "DATASET_MANIFEST", self.dataset),
            patch.object(pipeline, "RAW_DIR", self.raw),
            patch.object(pipeline, "SOURCE_MANIFEST", self.sources),
            patch.object(pipeline, "SOURCE_REVIEW", self.review),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        self.temp.cleanup()

    def write_dataset(self, allowed=True):
        self.dataset.write_text(
            json.dumps({"training_allowed": allowed}), encoding="utf-8"
        )

    def write_sources(self, *, allowed=True, with_url=False):
        source_file = self.raw / "sample.jsonl"
        record = {"text": "Türkçe örnek metin"}
        if with_url:
            record["url"] = "https://example.org/page"
        source_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
        source = {
            "source_id": "unit-test",
            "path": source_file.name,
            "license": "CC0",
            "permitted_uses": "AI training",
            "allow_training": allowed,
        }
        self.sources.write_text(
            json.dumps({"sources": [source]}), encoding="utf-8"
        )

    def write_approved_review(self, status="APPROVED"):
        columns = [
            "domain", "license_name", "license_evidence_url", "rights_holder",
            "ai_training_permission", "attribution_required", "review_status",
        ]
        with self.review.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({
                "domain": "example.org",
                "license_name": "CC0",
                "license_evidence_url": "https://example.org/license",
                "rights_holder": "Example Rights Holder",
                "ai_training_permission": "YES",
                "attribution_required": "NO",
                "review_status": status,
            })

    def test_missing_dataset_manifest_blocks(self):
        ok, reason = pipeline.approval_status()
        self.assertFalse(ok)
        self.assertIn("Missing dataset manifest", reason)

    def test_dataset_false_blocks(self):
        self.write_dataset(False)
        self.write_sources()
        ok, reason = pipeline.approval_status()
        self.assertFalse(ok)
        self.assertIn("Boolean true", reason)

    def test_string_true_does_not_approve(self):
        self.write_dataset("true")
        self.write_sources()
        ok, _ = pipeline.approval_status()
        self.assertFalse(ok)

    def test_unapproved_source_blocks(self):
        self.write_dataset(True)
        self.write_sources(allowed=False)
        ok, reason = pipeline.approval_status()
        self.assertFalse(ok)
        self.assertIn("Source not approved", reason)

    def test_approved_non_url_source_passes(self):
        self.write_dataset(True)
        self.write_sources(allowed=True, with_url=False)
        ok, reason = pipeline.approval_status()
        self.assertTrue(ok, reason)

    def test_url_source_requires_approved_domain_review(self):
        self.write_dataset(True)
        self.write_sources(allowed=True, with_url=True)
        ok, reason = pipeline.approval_status()
        self.assertFalse(ok)
        self.assertIn("domain review CSV", reason)

        self.write_approved_review(status="PENDING")
        ok, reason = pipeline.approval_status()
        self.assertFalse(ok)
        self.assertIn("pending or incomplete", reason)

        self.write_approved_review(status="APPROVED")
        ok, reason = pipeline.approval_status()
        self.assertTrue(ok, reason)

    def test_pending_template_is_created_and_never_auto_approved(self):
        self.write_sources(allowed=False, with_url=True)
        pipeline.create_pending_domain_template()
        with self.review.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "example.org")
        self.assertEqual(rows[0]["review_status"], "PENDING")
        self.assertEqual(rows[0]["ai_training_permission"], "NO")

    def test_existing_review_is_never_overwritten(self):
        self.write_sources(allowed=False, with_url=True)
        self.review.write_text("human-reviewed\n", encoding="utf-8")
        pipeline.create_pending_domain_template()
        self.assertEqual(self.review.read_text(encoding="utf-8"), "human-reviewed\n")


if __name__ == "__main__":
    unittest.main()
