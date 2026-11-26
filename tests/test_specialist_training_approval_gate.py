import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ci.trigger_space_training import validate_training_approvals


def write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def governance(model="code"):
    return {
        "schema": "aethrion-human-review-v1",
        "model_id": model,
        "approval_status": "approved",
        "author_role": "content creator",
        "authorship_attested": True,
        "first_party_origin_attested": True,
        "rights_holder_role": "rights holder",
        "rights_attested": True,
        "source_rights_reviewed": True,
        "privacy_reviewed": True,
        "quality_reviewed": True,
        "reviewer_role": "independent reviewer",
        "reviewed_at": "2026-11-25T12:00:00+00:00",
        "evidence_uri": "review-record://fixture",
        "speaker_consent_reviewed": False,
        "safety_reviewed": False,
    }


class SpecialistTrainingApprovalGateTests(unittest.TestCase):
    def build_fixture(self, root, *, approved=True):
        source = b'{"id":"fixture-1"}\\n'
        dataset = b"fixture dataset"
        tokenizer = b'{"vocab":{}}'
        source_hash = write(root / "data/specialists/manifests/code/source.jsonl", source)
        dataset_hash = write(root / "data/specialists/manifests/code/dataset.npz", dataset)
        tokenizer_hash = write(root / "data/specialists/manifests/code/tokenizer.json", tokenizer)
        record = governance()
        if not approved:
            record["approval_status"] = "pending"
        (root / "data/authored/code").mkdir(parents=True, exist_ok=True)
        (root / "data/authored/code/GOVERNANCE.json").write_text(json.dumps(record), encoding="utf-8")
        manifest = {
            "model_id": "code",
            "training_allowed": approved,
            "owner_approval": approved,
            "privacy_reviewed": approved,
            "dataset_path": "code/dataset.npz",
            "dataset_sha256": dataset_hash,
            "tokenizer_path": "code/tokenizer.json",
            "tokenizer_sha256": tokenizer_hash,
            "sources": [{
                "source_id": "fixture",
                "source_path": "code/source.jsonl",
                "sha256": source_hash,
                "allow_training": approved,
                "privacy_reviewed": approved,
            }],
            "governance": record,
        }
        path = root / "data/specialists/manifests/code.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_pending_review_blocks_before_any_network_client_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_fixture(root, approved=False)
            with self.assertRaisesRegex(RuntimeError, "approval"):
                validate_training_approvals(["code"], root)

    def test_valid_fixture_passes_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_fixture(root, approved=True)
            validate_training_approvals(["code"], root)

    def test_source_hash_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_fixture(root, approved=True)
            (root / "data/specialists/manifests/code/source.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_training_approvals(["code"], root)

    def test_source_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_fixture(root, approved=True)
            manifest_path = root / "data/specialists/manifests/code.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["source_path"] = "../../authored/code/GOVERNANCE.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                validate_training_approvals(["code"], root)

    def test_dataset_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_fixture(root, approved=True)
            manifest_path = root / "data/specialists/manifests/code.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_path"] = "../../authored/code/GOVERNANCE.json"
            manifest["dataset_sha256"] = hashlib.sha256((root / "data/authored/code/GOVERNANCE.json").read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                validate_training_approvals(["code"], root)

    def test_unsupported_family_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            validate_training_approvals(["chat"], Path("."))


if __name__ == "__main__":
    unittest.main()
