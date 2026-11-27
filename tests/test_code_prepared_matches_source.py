import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CodePreparedMatchesSourceTests(unittest.TestCase):
    def test_prepared_source_equals_authored_and_hashes_match(self):
        authored_path = ROOT / "data/authored/code/records.jsonl"
        authored = [json.loads(line) for line in authored_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prep = ROOT / "data/specialists/manifests/code"
        prepared = [json.loads(line) for line in (prep / "source.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(
            [(row["id"], row["group_id"], row["text"]) for row in authored],
            [(row["id"], row["group_id"], row["text"]) for row in prepared],
        )
        manifest = json.loads((ROOT / "data/specialists/manifests/code.json").read_text(encoding="utf-8"))
        governance = json.loads((ROOT / "data/authored/code/GOVERNANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((prep / "dataset.npz").read_bytes()).hexdigest(), manifest["dataset_sha256"])
        self.assertEqual(hashlib.sha256((prep / "tokenizer.json").read_bytes()).hexdigest(), manifest["tokenizer_sha256"])
        self.assertEqual(hashlib.sha256(authored_path.read_bytes()).hexdigest(), manifest["sources"][0]["sha256"])
        self.assertEqual(manifest["governance"], governance)
        self.assertEqual(
            manifest["training_allowed"],
            governance["approval_status"] == "approved"
            and governance["rights_attested"] is True
            and governance["privacy_reviewed"] is True,
        )
        self.assertEqual(manifest["owner_approval"], governance["approval_status"] == "approved")
        self.assertTrue(manifest["privacy_reviewed"])
        self.assertTrue(governance["quality_reviewed"])


if __name__ == "__main__":
    unittest.main()
