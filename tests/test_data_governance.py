import unittest

from aethrion.data_governance import (
    GovernanceError,
    review_template,
    validate_governance_record,
)


def approved_record(model_id="base"):
    record = review_template(model_id)
    record.update({
        "approval_status": "approved",
        "author_role": "project contributor",
        "authorship_attested": True,
        "rights_holder_role": "project contributor",
        "rights_attested": True,
        "source_rights_reviewed": True,
        "privacy_reviewed": True,
        "quality_reviewed": True,
        "reviewer_role": "independent human reviewer",
        "reviewed_at": "2026-11-18T10:00:00+03:00",
        "evidence_uri": "review-record://dataset-001",
    })
    return record


class DatasetGovernanceTests(unittest.TestCase):
    def test_template_is_never_approved(self):
        template = review_template("base")
        self.assertEqual(template["approval_status"], "pending")
        self.assertFalse(template["authorship_attested"])
        with self.assertRaisesRegex(GovernanceError, "not approved"):
            validate_governance_record(template, model_id="base")

    def test_complete_timestamped_human_review_is_accepted(self):
        record = approved_record()
        self.assertEqual(validate_governance_record(record, model_id="base"), record)

    def test_model_identity_must_match(self):
        with self.assertRaisesRegex(GovernanceError, "model_id"):
            validate_governance_record(approved_record("code"), model_id="base")

    def test_missing_authorship_attestation_blocks(self):
        record = approved_record()
        record["authorship_attested"] = False
        with self.assertRaisesRegex(GovernanceError, "authorship_attested"):
            validate_governance_record(record, model_id="base")

    def test_timezone_is_required(self):
        record = approved_record()
        record["reviewed_at"] = "2026-11-18T10:00:00"
        with self.assertRaisesRegex(GovernanceError, "timezone"):
            validate_governance_record(record, model_id="base")

    def test_audio_requires_speaker_consent_review(self):
        record = approved_record("asr")
        with self.assertRaisesRegex(GovernanceError, "speaker-consent"):
            validate_governance_record(record, model_id="asr")
        record["speaker_consent_reviewed"] = True
        validate_governance_record(record, model_id="asr")

    def test_safe_requires_separate_safety_review(self):
        record = approved_record("safety")
        with self.assertRaisesRegex(GovernanceError, "SAFE data"):
            validate_governance_record(record, model_id="safety")
        record["safety_reviewed"] = True
        validate_governance_record(record, model_id="safety")


if __name__ == "__main__":
    unittest.main()
