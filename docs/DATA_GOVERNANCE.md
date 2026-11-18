# AETHRION dataset governance and human review

Training approval is a separate human decision. BASE and specialist data-preparation commands may validate, normalize, hash, and split owner-supplied records while approvals are pending. This preparation does not authorize training; all training flags remain closed until the independent approval gate passes.

## Create a review template

Create a pending template for the dataset being reviewed:

```bash
python -m aethrion.data_governance --model base --output data/reviews/base-review.json
python -m aethrion.data_governance --model safety --output data/reviews/safety-review.json
python -m aethrion.data_governance --model asr --output data/reviews/asr-review.json
```

Supported model IDs: `base`, `chat`, `code`, `vision`, `image`, `asr`, `tts`, `vocoder`, `3d`, `research`, `document`, `safety`.

## Required review record

The dataset manifest must contain a `governance` object using schema `aethrion-human-review-v1`. It records:

- the model ID and explicit `approval_status: approved`;
- author and rights-holder roles, with authorship and rights attestations;
- source-rights, privacy, and quality review;
- a reviewer role, timezone-aware ISO-8601 review timestamp, and retained evidence reference;
- speaker-consent review for ASR/TTS/VOCODER;
- a separate safety-data review for SAFE.

Use roles rather than legal names, account IDs, addresses, or other personal identifiers. Audio recordings must have explicit speaker permission. Do not include private conversations, secrets, or identifying personal data in corpora.

## Important limitation

These fields are auditable attestations, not cryptographic proof that a person owns content or that consent was genuinely obtained. A responsible human must inspect the underlying source, rights evidence, privacy impact, quality, and any required consent before changing the template to approved. Keep the review evidence under controlled access. The validator rejects missing, pending, mismatched, or incomplete attestations; it does not manufacture approval.

Even after governance passes, the existing source hashes, dataset integrity checks, group-disjoint train/validation/test splits, modality checks, training reports, evaluation benchmarks, safety gates, and release review remain required. For BASE, run corpus preparation with the dataset and source manifests still marked pending; only the separate `validate_training_approval` step may authorize a training run.
