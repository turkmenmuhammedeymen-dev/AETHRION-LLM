import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CodePreparedMatchesSourceTests(unittest.TestCase):
    def test_prepared_source_equals_authored_and_hashes_match(self):
        authored = [json.loads(l) for l in (ROOT / "data/authored/code/records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        prep = ROOT / "data/specialists/manifests/code"
        prepared = [json.loads(l) for l in (prep / "source.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual([(r["id"], r["group_id"], r["text"]) for r in authored],
                         [(r["id"], r["group_id"], r["text"]) for r in prepared])
        manifest = json.loads((ROOT / "data/specialists/manifests/code.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((prep / "dataset.npz").read_bytes()).hexdigest(), manifest["dataset_sha256"])
        self.assertEqual(hashlib.sha256((prep / "tokenizer.json").read_bytes()).hexdigest(), manifest["tokenizer_sha256"])
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(manifest["owner_approval"])


if __name__ == "__main__":
    unittest.main()
