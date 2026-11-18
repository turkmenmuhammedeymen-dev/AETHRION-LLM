import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aethrion.research_index import ResearchIndexError, build_local_research_index
from aethrion.tokenizer import ByteBPETokenizer


class NativeResearchIndexTests(unittest.TestCase):
    def _fixture(self, root: Path, approved=True):
        source = root / "source.txt"
        source_text = "AETHRION locally indexed research source with evidence and provenance. " * 8
        source.write_text(source_text, encoding="utf-8")
        tokenizer_path = root / "tokenizer.json"
        tokenizer = ByteBPETokenizer()
        tokenizer.train(source_text, vocab_size=256)
        tokenizer.save(tokenizer_path)
        records = root / "records.jsonl"
        records.write_text(json.dumps({
            "doc_id": "doc-1", "source_id": "source-1",
            "title": "Local source title", "text": source_text
        }) + "\n", encoding="utf-8")
        approval = root / "approval.json"
        approval.write_text(json.dumps({
            "format": "aethrion-research-index-approval-v1",
            "index_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "records_path": "records.jsonl",
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "sources": [{
                "source_id": "source-1",
                "source_path": "source.txt",
                "source_uri": "https://example.invalid/source",
                "license": "Test fixture license",
                "permitted_use": "Unit tests only",
                "evidence_uri": "unit-test://evidence",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "allow_index": approved,
                "privacy_reviewed": approved
            }]
        }), encoding="utf-8")
        return records, approval, tokenizer_path

    def test_index_is_local_and_provenance_preserving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, approval, tokenizer = self._fixture(root)
            processor = build_local_research_index(
                records, approval, tokenizer, root / "artifacts" / "research",
                chunk_tokens=64, overlap_tokens=8, project_root=root
            )
            self.assertEqual(processor["execution_policy"], "local_only")
            index_path = root / processor["index_path"]
            metadata_path = root / processor["metadata_path"]
            self.assertEqual(hashlib.sha256(index_path.read_bytes()).hexdigest(), processor["index_sha256"])
            with np.load(index_path, allow_pickle=False) as archive:
                self.assertEqual(archive["passage_ids"].ndim, 2)
                self.assertEqual(archive["passage_ids"].shape, archive["passage_mask"].shape)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata[0]["source_uri"], "https://example.invalid/source")
            self.assertEqual(metadata[0]["license"], "Test fixture license")

    def test_records_jsonl_must_match_approved_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, approval, tokenizer = self._fixture(root)
            records.write_text('{"doc_id":"tampered"}\\n', encoding="utf-8")
            with self.assertRaisesRegex(ResearchIndexError, "records JSONL SHA-256"):
                build_local_research_index(
                    records, approval, tokenizer, root / "artifacts" / "research",
                    chunk_tokens=64, project_root=root
                )


    def test_symlinked_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, approval, tokenizer = self._fixture(root)
            source = root / "source.txt"
            real_source = root / "real-source.txt"
            source.rename(real_source)
            try:
                source.symlink_to(real_source.name)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic links are unavailable in this environment.")
            with self.assertRaisesRegex(ResearchIndexError, "symbolic links"):
                build_local_research_index(
                    records, approval, tokenizer, root / "artifacts" / "research",
                    chunk_tokens=64, project_root=root
                )

    def test_symlinked_records_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, approval, tokenizer = self._fixture(root)
            real_records = root / "real-records.jsonl"
            records.rename(real_records)
            try:
                records.symlink_to(real_records.name)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic links are unavailable in this environment.")
            with self.assertRaisesRegex(ResearchIndexError, "symbolic links"):
                build_local_research_index(
                    records, approval, tokenizer, root / "artifacts" / "research",
                    chunk_tokens=64, project_root=root
                )

    def test_unapproved_index_use_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, approval, tokenizer = self._fixture(root, approved=False)
            with self.assertRaisesRegex(ResearchIndexError, "index_allowed"):
                build_local_research_index(
                    records, approval, tokenizer, root / "artifacts" / "research",
                    chunk_tokens=64, project_root=root
                )


if __name__ == "__main__":
    unittest.main()
