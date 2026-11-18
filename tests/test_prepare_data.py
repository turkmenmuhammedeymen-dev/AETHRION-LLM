import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["AETHRION_BASE_AUDIT_HMAC_KEY"] = "unit-test-only-key-not-for-production-0001"

from aethrion.prepare_data import (
    build_corpus,
    load_sources_manifest,
    load_domain_review,
    normalize_text,
    require_training_enabled,
    validate_training_approval,
    _write_report_signature,
    _verify_report_signature,
)


class CorpusPreparationTests(unittest.TestCase):
    def _write_dataset_manifest(self, root: Path, allowed: bool = True) -> Path:
        path = root / "corpus_manifest.json"
        path.write_text(
            json.dumps({
                "project": "AETHRION-LLM-tests",
                "training_allowed": allowed,
                "governance": {
                    "schema": "aethrion-human-review-v1",
                    "model_id": "base",
                    "approval_status": "approved",
                    "author_role": "unit-test author",
                    "authorship_attested": True,
                    "first_party_origin_attested": True,
                    "first_party_origin_attested": True,
                    "rights_holder_role": "unit-test rights holder",
                    "rights_attested": True,
                    "source_rights_reviewed": True,
                    "privacy_reviewed": True,
                    "quality_reviewed": True,
                    "reviewer_role": "unit-test reviewer",
                    "reviewed_at": "2026-01-01T00:00:00Z",
                    "evidence_uri": "unit-test://human-review",
                },
            }),
            encoding="utf-8",
        )
        return path

    def test_unicode_and_whitespace(self):
        self.assertEqual(normalize_text("  Türkçe\n  metin  "), "Türkçe metin")

    def test_manifest_pipeline_deduplicates_and_filters_email(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = "Bu, AETHRION için yeterince uzun bir Türkçe eğitim cümlesidir."
            (root / "source-a.txt").write_text((first + "\n") * 2, encoding="utf-8")
            (root / "source-b.txt").write_text(
                "İkinci ve farklı bir Türkçe eğitim örneği burada bulunuyor.\n"
                + "İletişim test@example.com olan kayıt dışarıda kalmalıdır.\n",
                encoding="utf-8",
            )
            (root / "source-c.txt").write_text(
                "Üçüncü ve bağımsız bir Türkçe eğitim örneği de burada bulunuyor.\n",
                encoding="utf-8",
            )
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [
                {"source_id": "unit-a", "path": "source-a.txt",
                 "license": "Synthetic unit-test fixture",
                 "permitted_uses": "training permitted for unit-test fixture only",
                 "allow_training": False},
                {"source_id": "unit-b", "path": "source-b.txt",
                 "license": "Synthetic unit-test fixture",
                 "permitted_uses": "training permitted for unit-test fixture only",
                 "allow_training": False},
                {"source_id": "unit-c", "path": "source-c.txt",
                 "license": "Synthetic unit-test fixture",
                 "permitted_uses": "training permitted for unit-test fixture only",
                 "allow_training": False},
            ]}), encoding="utf-8")
            output = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            report = build_corpus(
                root, manifest, output, report_path, seed=7, min_chars=10,
                min_records=1, dataset_manifest_path=self._write_dataset_manifest(root, allowed=False),
            )
            self.assertEqual(report["totals"]["records_kept"], 3)
            self.assertEqual(report["totals"]["duplicates"], 1)
            self.assertEqual(report["totals"]["email_like_records_excluded"], 1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(set(report["splits"]), {"train", "validation", "test"})
            self.assertTrue(all(Path(item["path"]).is_file() for item in report["splits"].values()))
            self.assertTrue(report_path.exists())

    def test_tampered_split_is_rejected_even_when_report_hashes_are_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "sources.json"
            sources = []
            for index in range(3):
                path = root / f"source-{index}.txt"
                path.write_text(
                    (f"Independent reviewed source group {index} contains this training sentence.\\n") * 40,
                    encoding="utf-8",
                )
                sources.append({
                    "source_id": f"source-{index}",
                    "path": path.name,
                    "license": "Synthetic unit-test fixture",
                    "permitted_uses": "training permitted for unit-test fixture only",
                    "allow_training": True,
                })
            manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out" / "corpus.txt"
            report_path = root / "out" / "report.json"
            report = build_corpus(
                root, manifest, corpus, report_path, min_chars=10, min_records=1,
                dataset_manifest_path=dataset_manifest,
            )
            train_path = Path(report["splits"]["train"]["path"])
            lines = train_path.read_text(encoding="utf-8").splitlines()
            lines[0] = "Tampered replacement text not present in the reviewed raw source."
            train_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
            for split_name in ("train", "validation", "test"):
                split_path = Path(report["splits"][split_name]["path"])
                raw = split_path.read_bytes()
                report["splits"][split_name]["sha256"] = hashlib.sha256(raw).hexdigest()
                report["splits"][split_name]["bytes"] = len(raw)
            combined = b"".join(
                Path(report["splits"][name]["path"]).read_bytes()
                for name in ("train", "validation", "test")
            )
            corpus.write_bytes(combined)
            report["output"]["sha256"] = hashlib.sha256(combined).hexdigest()
            report["output"]["bytes"] = len(combined)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                validate_training_approval(
                    dataset_manifest, manifest, root, corpus, report_path
                )

    def test_group_id_and_report_rewrite_attack_fails_hmac_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            source_paths = []
            for index in range(3):
                path = root / f"source-{index}.jsonl"
                path.write_text(json.dumps({
                    "group_id": f"reviewed-group-{index}",
                    "text": f"Unique independent source {index} provides enough text for this fixture.",
                }) + "\n", encoding="utf-8")
                source_paths.append(path)
                entries.append({
                    "source_id": f"source-{index}",
                    "path": path.name,
                    "license": "Synthetic unit-test fixture",
                    "permitted_uses": "training permitted for unit-test fixture only",
                    "allow_training": True,
                })
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            report = build_corpus(
                root, manifest, corpus, report_path, min_chars=10, min_records=1,
                dataset_manifest_path=dataset_manifest,
            )
            signature_path = Path(str(report_path) + ".hmac")
            self.assertTrue(signature_path.is_file())

            # Simulate an attacker relabeling provenance and rewriting the report's
            # source digest to match the changed raw source, while lacking the HMAC key.
            payload = json.loads(source_paths[0].read_text(encoding="utf-8"))
            payload["group_id"] = "attacker-invented-independent-group"
            source_paths[0].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report["sources"][0]["sha256"] = hashlib.sha256(
                source_paths[0].read_bytes()
            ).hexdigest()
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                validate_training_approval(
                    dataset_manifest, manifest, root, corpus, report_path
                )

    def test_signed_report_is_bound_to_current_source_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            for index in range(3):
                path = root / f"source-{index}.txt"
                path.write_text(
                    f"Independent source {index} contains a unique reviewed sentence.\n",
                    encoding="utf-8",
                )
                entries.append({
                    "source_id": f"source-{index}",
                    "path": path.name,
                    "license": "Synthetic unit-test fixture",
                    "permitted_uses": "training permitted for unit-test fixture only",
                    "allow_training": True,
                })
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            build_corpus(
                root, manifest, corpus, report_path, min_chars=10, min_records=1,
                dataset_manifest_path=dataset_manifest,
            )

            # A valid report signature cannot authorize a changed source manifest.
            entries[0]["license"] = "Changed after review"
            manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound to the current dataset"):
                validate_training_approval(
                    dataset_manifest, manifest, root, corpus, report_path
                )

    def test_explicit_group_ids_remain_together_across_source_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            for index in range(3):
                path = root / f"source-{index}.jsonl"
                rows = [
                    {
                        "group_id": "same-speaker",
                        "text": f"Shared speaker utterance number {index} belongs to one group.",
                    },
                    {
                        "group_id": f"independent-author-{index}",
                        "text": f"Independent author {index} writes a different long sentence.",
                    },
                ]
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                entries.append({
                    "source_id": f"jsonl-source-{index}",
                    "path": path.name,
                    "license": "Synthetic unit-test fixture",
                    "permitted_uses": "training permitted for unit-test fixture only",
                    "allow_training": True,
                })
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out" / "corpus.txt"
            report_path = root / "out" / "report.json"
            report = build_corpus(
                root, manifest, corpus, report_path, min_chars=10, min_records=1,
                dataset_manifest_path=dataset_manifest,
            )
            shared_hash = hashlib.sha256(b"group:same-speaker").hexdigest()
            containing = [
                split_name for split_name, item in report["splits"].items()
                if shared_hash in item["groups"]
            ]
            self.assertEqual(len(containing), 1)
            validate_training_approval(
                dataset_manifest, manifest, root, corpus, report_path
            )

    def test_explicit_group_overrides_url_host_group_across_domains(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mixed.jsonl"
            rows = [
                {"url": "https://a.example.org/one", "group_id": "shared-author",
                 "text": "Shared author first source record has enough text."},
                {"url": "https://b.example.org/two", "group_id": "shared-author",
                 "text": "Shared author second domain record has enough words."},
                {"url": "https://c.example.org/three", "group_id": "independent-c",
                 "text": "Independent third author source record is separate."},
                {"url": "https://a.example.org/four", "group_id": "independent-d",
                 "text": "Another independent author contributes this record."},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [{
                "source_id": "mixed-reviewed",
                "path": source.name,
                "license": "CC0",
                "permitted_uses": "training permitted for unit-test fixture only",
                "allow_training": True,
            }]}), encoding="utf-8")
            review = root / "review.csv"
            columns = [
                "domain", "license_name", "license_evidence_url", "rights_holder",
                "ai_training_permission", "attribution_required", "review_status",
            ]
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for domain in ("a.example.org", "b.example.org", "c.example.org"):
                    writer.writerow({
                        "domain": domain,
                        "license_name": "CC0",
                        "license_evidence_url": f"https://{domain}/license",
                        "rights_holder": "Example rights holder",
                        "ai_training_permission": "YES",
                        "attribution_required": "NO",
                        "review_status": "APPROVED",
                    })
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out" / "corpus.txt"
            report_path = root / "out" / "report.json"
            report = build_corpus(
                root, manifest, corpus, report_path, min_chars=10, min_records=1,
                source_review_path=review, dataset_manifest_path=dataset_manifest,
            )
            shared_hash = hashlib.sha256(b"group:shared-author").hexdigest()
            split_memberships = [
                name for name, item in report["splits"].items()
                if shared_hash in item["groups"]
            ]
            self.assertEqual(len(split_memberships), 1)
            validate_training_approval(
                dataset_manifest, manifest, root, corpus, report_path, review
            )

    def test_source_manifest_rejects_duplicate_source_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("one.txt", "two.txt"):
                (root / name).write_text("A source file for identity validation.", encoding="utf-8")
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [
                {"source_id": "same-id", "path": "one.txt", "license": "Unit test",
                 "permitted_uses": "training", "allow_training": True},
                {"source_id": "same-id", "path": "two.txt", "license": "Unit test",
                 "permitted_uses": "training", "allow_training": True},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
                load_sources_manifest(manifest, root)

    def test_source_manifest_rejects_non_string_identity_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text("A source file for identity validation.", encoding="utf-8")
            for source_id, path in ((7, "source.txt"), ("valid-id", "../outside.txt"), ("valid-id", "https://example.org/data.txt")):
                manifest = root / "sources.json"
                manifest.write_text(json.dumps({"sources": [{
                    "source_id": source_id, "path": path, "license": "Unit test",
                    "permitted_uses": "training", "allow_training": True,
                }]}), encoding="utf-8")
                with self.subTest(source_id=source_id, path=path):
                    with self.assertRaises(ValueError):
                        load_sources_manifest(manifest, root)

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
        rows = [
            {"url": "https://a.example.org/article", "text": "Birinci kaynak alan adı için yeterince uzun eğitim metni."},
            {"url": "https://b.example.org/article", "text": "İkinci bağımsız alan adı için yeterince uzun metin."},
            {"url": "https://c.example.org/article", "text": "Üçüncü ayrı alan adı için yeterince uzun metin."},
        ]
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = root / "sources.json"
        manifest.write_text(json.dumps({"sources": [{
            "source_id": "mixed-test",
            "path": "mixed.jsonl",
            "license": "dataset-label",
            "permitted_uses": "training review required",
            "allow_training": True,
        }]}), encoding="utf-8")
        return source, manifest

    def test_url_records_can_be_prepared_but_training_requires_domain_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_url_source(root)
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            report = build_corpus(
                root, root / "sources.json", corpus,
                report_path, min_chars=10, min_records=1,
                dataset_manifest_path=dataset_manifest,
            )
            self.assertEqual(
                {item["review_status"] for item in report["domain_review"]},
                {"PENDING"},
            )
            with self.assertRaisesRegex(ValueError, "--source-review"):
                validate_training_approval(
                    dataset_manifest, root / "sources.json", root, corpus, report_path
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
            dataset_manifest = self._write_dataset_manifest(root)
            corpus = root / "out/corpus.txt"
            report_path = root / "out/report.json"
            build_corpus(
                root, root / "sources.json", corpus,
                report_path, min_chars=10, min_records=1,
                source_review_path=review,
                dataset_manifest_path=dataset_manifest,
            )
            with self.assertRaisesRegex(ValueError, "unapproved domains"):
                validate_training_approval(
                    dataset_manifest, root / "sources.json", root, corpus,
                    report_path, source_review_path=review,
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
                for domain in ("a.example.org", "b.example.org", "c.example.org"):
                    writer.writerow({
                        "domain": domain,
                        "license_name": "CC0",
                        "license_evidence_url": f"https://{domain}/license",
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
            self.assertEqual(
                {item["domain"] for item in report["domain_review"]},
                {"a.example.org", "b.example.org", "c.example.org"},
            )
            self.assertTrue(all(item["records"] == 1 for item in report["domain_review"]))


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

    def test_dataset_manifest_rejects_true_without_human_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unreviewed.json"
            path.write_text(json.dumps({
                "project": "AETHRION-LLM-tests",
                "training_allowed": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "human-review-v1"):
                require_training_enabled(path)

    def test_dataset_manifest_accepts_exact_boolean_true(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "approved.json"
            payload = {
                "project": "AETHRION-LLM-tests",
                "training_allowed": True,
                "governance": {
                    "schema": "aethrion-human-review-v1",
                    "model_id": "base",
                    "approval_status": "approved",
                    "author_role": "unit-test author",
                    "authorship_attested": True,
                    "first_party_origin_attested": True,
                    "rights_holder_role": "unit-test rights holder",
                    "rights_attested": True,
                    "source_rights_reviewed": True,
                    "privacy_reviewed": True,
                    "quality_reviewed": True,
                    "reviewer_role": "unit-test reviewer",
                    "reviewed_at": "2026-01-01T00:00:00Z",
                    "evidence_uri": "unit-test://human-review",
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = require_training_enabled(path)
            self.assertIs(result["training_allowed"], True)


    def test_hmac_sidecar_requires_configured_key_and_valid_format(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            report.write_text('{"format":"test"}', encoding="utf-8")
            key_name = "AETHRION_BASE_AUDIT_HMAC_KEY"
            previous = os.environ.pop(key_name, None)
            try:
                with self.assertRaisesRegex(ValueError, "key is not configured"):
                    _verify_report_signature(report)
            finally:
                if previous is not None:
                    os.environ[key_name] = previous
            sidecar = Path(str(report) + ".hmac")
            sidecar.write_text("not-a-valid-signature\\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "malformed"):
                _verify_report_signature(report)

    def test_hmac_sidecar_rejects_symlinks_without_touching_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "report.json"
            report.write_text('{"format":"test"}', encoding="utf-8")
            victim = root / "victim.txt"
            victim.write_text("must remain unchanged", encoding="utf-8")
            sidecar = Path(str(report) + ".hmac")
            try:
                sidecar.symlink_to(victim)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation is unavailable on this platform")
            with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                _write_report_signature(report)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                _verify_report_signature(report)
            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged")

    def test_hmac_sidecar_write_is_atomic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            report.write_text('{"format":"test","value":1}', encoding="utf-8")
            self.assertTrue(_write_report_signature(report))
            sidecar = Path(str(report) + ".hmac")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(list(report.parent.glob(".report.json.hmac.*.tmp")), [])
            _verify_report_signature(report)
            report.write_text('{"format":"test","value":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                _verify_report_signature(report)



    def test_source_manifest_rejects_disguised_pending_license_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text("Owner-authored test source.", encoding="utf-8")
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"sources": [{
                "source_id": "owner-source",
                "path": "source.txt",
                "license": "PENDING - awaiting human review",
                "permitted_uses": "AI training",
                "allow_training": True,
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "license is not verified"):
                load_sources_manifest(manifest, root, require_training_approval=True)

    def test_source_manifest_rejects_negative_or_ambiguous_training_permission(self):
        for permitted_uses in (
            "Not for training; evaluation only",
            "Not a production training corpus",
            "Not permitted for training",
        ):
            with self.subTest(permitted_uses=permitted_uses):
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    (root / "source.txt").write_text("Owner-authored test source.", encoding="utf-8")
                    manifest = root / "sources.json"
                    manifest.write_text(json.dumps({"sources": [{
                        "source_id": "owner-source",
                        "path": "source.txt",
                        "license": "AETHRION owner-authored",
                        "permitted_uses": permitted_uses,
                        "allow_training": True,
                    }]}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must explicitly include training"):
                        load_sources_manifest(manifest, root, require_training_approval=True)

    def test_domain_review_rejects_disguised_unknown_license_and_rights_holder(self):
        with tempfile.TemporaryDirectory() as folder:
            review = Path(folder) / "review.csv"
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "domain", "license_name", "license_evidence_url", "rights_holder",
                    "ai_training_permission", "attribution_required", "review_status",
                ])
                writer.writeheader()
                writer.writerow({
                    "domain": "example.org",
                    "license_name": "UNKNOWN - pending",
                    "license_evidence_url": "https://example.org/license",
                    "rights_holder": "UNKNOWN - not checked",
                    "ai_training_permission": "YES",
                    "attribution_required": "NO",
                    "review_status": "APPROVED",
                })
            result = load_domain_review(review)
            self.assertFalse(result["example.org"]["approved"])


if __name__ == "__main__":
    unittest.main()
