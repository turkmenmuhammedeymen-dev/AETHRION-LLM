# AETHRION specialist data onboarding

This document defines the production-data contract for the ten specialist models.
The manifests in `data/specialists/manifests/` are registrations only. They
intentionally contain no source records, prepared arrays, approvals, or training
permission. Do not interpret a registered manifest as an available dataset.

## First-party source requirements

- Use owner-authored material only. Third-party corpora and pretrained weights are
  not approved by this repository's current data policy.
- Record each source's stable ID, local source path, provenance URI, exact license
  or rights basis, explicit training permission, evidence URI, and SHA-256.
- Keep source files local to the repository and hash-verified.
- Complete timestamped human authorship, rights-holder, privacy, quality, and
  source-rights review before changing any approval field.
- Deduplicate inputs before creating group-disjoint train, validation, and test
  splits. Never place records from the same provenance group in multiple splits.
- Audio models additionally require documented speaker consent for every source.
- SAFE requires a separate human safety-data review. Labels must be reviewed by
  qualified human reviewers; synthetic smoke-test labels are not safety training
  evidence.

## Prepared NPZ contracts

All arrays use numeric dtypes and have a common sample axis. Every archive must
also contain unique `sample_id`, pseudonymous SHA-256 `group_id`, and `split`
arrays. Splits must include train, validation, and test.

| Model | Required arrays |
|---|---|
| CODE | `token_ids` |
| VISION | `images`, `token_ids` |
| IMAGE | `images`, `token_ids` |
| ASR | `features`, `input_lengths`, `targets`, `target_lengths` |
| TTS | `token_ids`, `token_mask`, `token_lengths`, `frame_to_token`, `frame_mask`, `frame_lengths`, `target_mels`, `durations` |
| VOCODER | `target_mels`, `target_waveforms`, `frame_lengths`, `waveform_mask` |
| 3D | `points`, `token_ids` |
| RESEARCH | `query_ids`, `passage_ids` |
| DOCUMENT | `token_ids`, `boxes`, `layout_ids`, `tag_labels` |
| SAFE | `token_ids`, `decision_labels`, `reason_labels` |

The authoritative shape, value-range, duplicate-input, split-leakage, source
hash, and approval checks live in `aethrion/specialist_data.py`. Do not bypass
those validators or lower their requirements to make a dataset pass.

## SAFE label policy

SAFE decision labels are `ALLOW=0`, `REVIEW=1`, and `BLOCK=2`. Reason labels
are binary multi-hot vectors. Each decision class must meet the configured
minimum of 25 reviewed examples in each required split. This minimum is a
floor, not proof of quality: class balance, ambiguity handling, reviewer
agreement, held-out performance, and adversarial evaluation still require
separate review. Do not invent or auto-label safety examples to fill a quota.

## Training gate

A specialist can enter training only when all of the following are true:

1. The manifest targets the intended model and modality.
2. `training_allowed` and `owner_approval` are explicit JSON booleans set to true.
3. The governance record passes `aethrion.data_governance.validate_governance_record`.
4. Every source is first-party, local, non-empty, hash-matched, and explicitly
   approved for model training with rights and privacy evidence.
5. The prepared NPZ exists, its SHA-256 matches, and all array, shape, uniqueness,
   split, and provenance checks pass.
6. Audio consent and SAFE-specific independent human review are complete where
   applicable.

Until those conditions are met, the expected status is **blocked / not trained**.
A successful training run would still be **not released** until independent
held-out evaluation, security and safety review, artifact integrity checks, and
human release approval are complete.
