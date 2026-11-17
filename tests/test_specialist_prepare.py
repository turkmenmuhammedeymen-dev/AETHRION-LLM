import json
from pathlib import Path
import tempfile
import unittest

from aethrion.specialist_prepare import PreparationError, _split, prepare_text_dataset
from aethrion.specialist_data import validate_specialist_dataset, SpecialistDataError


class SpecialistPreparationTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_code_jsonl_creates_npz_tokenizer_and_fail_closed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "code.jsonl"
            rows = [{"id": f"code-{i}", "group_id": f"group-{i}", "text": f"def function_{i}(): return {i}"} for i in range(80)]
            self._write_jsonl(source, rows)
            result = prepare_text_dataset(
                source, model_id="code", output_root=root / "prepared",
                license_name="Unit test only", permitted_use="CI fixture only",
                evidence_uri="unit-test://source", source_uri="unit-test://dataset", max_length=32,
            )
            self.assertTrue(Path(result["dataset"]).is_file())
            self.assertTrue(Path(result["tokenizer"]).is_file())
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_allowed"])
            self.assertFalse(manifest["owner_approval"])
            self.assertFalse(manifest["sources"][0]["allow_training"])
            self.assertEqual(sum(result["split_counts"].values()), 80)
            with self.assertRaises(SpecialistDataError):
                validate_specialist_dataset(result["manifest"])

    def test_chat_instruction_preparation_masks_prompt_and_pads_response_only_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chat.jsonl"
            rows = [
                {
                    "id": f"chat-{index}",
                    "group_id": f"conversation-group-{index}",
                    "prompt": f"Türkçe kullanıcı sorusu {index} nedir?",
                    "response": f"Bu, doğrulanmış test yanıtı {index} örneğidir.",
                }
                for index in range(90)
            ]
            self._write_jsonl(source, rows)
            result = prepare_text_dataset(
                source, model_id="chat", output_root=root / "prepared",
                license_name="Unit test only", permitted_use="CI fixture only",
                evidence_uri="unit-test://source", source_uri="unit-test://dataset",
                max_length=64,
            )
            with np.load(result["dataset"], allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"sample_id", "group_id", "split", "input_ids", "labels"})
                for labels in arrays["labels"]:
                    supervised = np.flatnonzero(labels != -100)
                    self.assertGreater(len(supervised), 0)
                    self.assertTrue(np.all(labels[:supervised[0]] == -100))
                    self.assertTrue(np.all(labels[supervised[-1] + 1:] == -100))
                manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
                self.assertEqual(manifest["modality"], "text")
                self.assertFalse(manifest["training_allowed"])
                self.assertFalse(manifest["owner_approval"])
                self.assertEqual(sum(result["split_counts"].values()), len(rows))

    def test_document_rejects_misaligned_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "document.jsonl"
            self._write_jsonl(source, [
                {"id": f"doc-{i}", "group_id": f"group-{i}", "text": "AB", "boxes": [[0, 0, 1, 1]],
                 "layout_ids": [0, 0], "tag_labels": [0, 0]}
                for i in range(80)
            ])
            with self.assertRaisesRegex(PreparationError, "match its token count"):
                prepare_text_dataset(
                    source, model_id="document", output_root=root / "prepared",
                    license_name="Unit test only", permitted_use="CI fixture only",
                    evidence_uri="unit-test://source", source_uri="unit-test://dataset", max_length=16,
                )

    def test_safety_accepts_balanced_manual_labels_but_keeps_approval_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "safe.jsonl"
            rows = []
            index = 0
            decisions = {"train": "ALLOW", "validation": "REVIEW", "test": "BLOCK"}
            for split, decision in decisions.items():
                for _ in range(25):
                    while _split(f"safe-{index}", 1337) != split:
                        index += 1
                    rows.append({"id": f"safe-{index}", "group_id": f"safe-{index}", "text": "Benign policy example.",
                                 "decision": decision, "reasons": [0]})
                    index += 1
            # Add the other two labels to every split so all three decisions are balanced.
            for split in ("train", "validation", "test"):
                for decision in ("ALLOW", "REVIEW", "BLOCK"):
                    for _ in range(25):
                        while _split(f"safe-{index}", 1337) != split:
                            index += 1
                        rows.append({"id": f"safe-{index}", "group_id": f"safe-{index}", "text": "Benign policy example.",
                                     "decision": decision, "reasons": [0]})
                        index += 1
            self._write_jsonl(source, rows)
            result = prepare_text_dataset(
                source, model_id="safety", output_root=root / "prepared",
                license_name="Unit test only", permitted_use="CI fixture only",
                evidence_uri="unit-test://source", source_uri="unit-test://dataset", max_length=16,
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_allowed"])
            self.assertEqual(manifest["safety_min_per_class"], 25)

    def test_safety_requires_balanced_manual_labels_in_every_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "safe.jsonl"
            rows = []
            index = 0
            for split in ("train", "validation", "test"):
                label = {"train": "ALLOW", "validation": "REVIEW", "test": "BLOCK"}[split]
                for _ in range(30):
                    while _split(f"safe-{index}", 1337) != split:
                        index += 1
                    rows.append({"id": f"safe-{index}", "text": "Benign policy example.",
                                 "decision": label, "reasons": [0]})
                    index += 1
            self._write_jsonl(source, rows)
            with self.assertRaisesRegex(PreparationError, "requires at least 25"):
                prepare_text_dataset(
                    source, model_id="safety", output_root=root / "prepared",
                    license_name="Unit test only", permitted_use="CI fixture only",
                    evidence_uri="unit-test://source", source_uri="unit-test://dataset", max_length=16,
                )


if __name__ == "__main__":
    unittest.main()
