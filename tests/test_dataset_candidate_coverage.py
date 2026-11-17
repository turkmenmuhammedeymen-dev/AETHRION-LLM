import json
import unittest
from pathlib import Path

MODEL_IDS = {
    "chat", "code", "vision", "image", "asr", "tts",
    "3d", "research", "document", "safety", "vocoder",
}


class DatasetCandidateCoverageTests(unittest.TestCase):
    def test_every_model_has_a_candidate_or_internal_authoring_track(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "data" / "dataset_candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "aethrion-dataset-candidate-register-v1")
        covered = {
            model_id
            for candidate in payload["candidates"]
            for model_id in candidate.get("model_ids", [])
        }
        self.assertEqual(covered, MODEL_IDS)

    def test_catalogue_never_implicitly_approves_or_downloads_data(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "data" / "dataset_candidates.json").read_text(encoding="utf-8"))
        self.assertIs(payload["training_approval_granted"], False)
        for candidate in payload["candidates"]:
            self.assertIs(candidate["training_allowed"], False)
            self.assertIs(candidate["downloaded"], False)

    def test_project_training_authorization_does_not_accept_provider_terms(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "data" / "dataset_candidates.json").read_text(encoding="utf-8"))
        approval = payload["owner_approval"]
        self.assertIs(approval["permits_training"], True)
        self.assertIs(approval["permits_accepting_provider_legal_terms"], False)
        self.assertIs(approval["permits_paid_acquisition"], False)
        self.assertIs(approval["permits_paid_compute"], False)


if __name__ == "__main__":
    unittest.main()
