# Owner-authored training data intake

AETHRION-LLM has modality-specific preparation code and fail-closed training gates. The missing prerequisite is real data with documented origin, rights, privacy, quality review, and (where relevant) speaker consent. This intake kit creates an empty local workspace; it does not create examples, approve sources, change production manifests, or start training.

## Create the workspace

From the repository root run:

```bash
python scripts/create_owner_data_intake.py
```

It creates data/intake/ with one folder per model. That path is ignored by Git. Existing paths are never overwritten; use --output to choose another new local directory.

Each model folder contains an empty records.jsonl, a submission.json checklist with approvals disabled, a governance.json human-review template in pending state, and modality asset folders where needed.

## Model-specific raw record fields

| Model | Required fields | Additional review |
|---|---|---|
| CHAT | id, group_id, prompt, response | Reuse exact BASE tokenizer; no private chat logs. |
| CODE | id, group_id, text | Original code and dependency/license provenance. |
| VISION | id, group_id, image_path, caption | Image origin, training rights, privacy and subject consent. |
| IMAGE | id, group_id, image_path, caption | Rights-cleared image-caption pairs; privacy and memorization review. |
| ASR | id, speaker_id, wav_path, transcript | Local WAV and documented speaker consent. |
| TTS | id, speaker_id, wav_path, transcript, durations | WAV, verified token/frame alignments and speaker consent. |
| VOCODER | id, speaker_id, wav_path | WAV and speaker consent. |
| 3D | id, group_id, point_path, text | Point-cloud ownership and scan/CAD provenance. |
| RESEARCH | id, group_id, query, passage | First-party passages or explicit training permission; citations. |
| DOCUMENT | id, group_id, text, boxes, layout_ids, tag_labels | Rights-cleared documents; exclude personal/confidential data. |
| SAFETY | id, group_id, text, decision, reasons | Human-assigned labels and separate safety-data review. |

The existing data/specialists/README.md and preparation modules remain authoritative for tensor shapes, preprocessing, modality constraints, and CLI options.

## Evidence required before approval

1. Origin/authorship: creator, role, and explicit first-party origin attestation for the complete dataset.
2. Rights: actual rights holder, license or written permission, and explicit permission for AI/model training.
3. Privacy: remove personal/confidential information unless a responsible reviewer documents a lawful basis and required consent.
4. Quality: human review of labels, modality, alignment, duplicates, provenance and limitations.
5. Evaluation leakage: unique sample IDs and group-disjoint train/validation/test partitions. The same speaker, repository, document or source group must stay in one split.
6. Audio: ASR, TTS and Vocoder each require explicit speaker consent covering model training; consent for one purpose does not imply another.
7. SAFE: human-reviewed decision/reason labels and separate safety review. Use benign prevention-oriented examples, not operational harmful instructions.

A responsible adult should support rights, privacy and consent review. Do not publish private messages, school records, medical records, account details or identifiable recordings into the repository.

## Preparation and final gate

Use the existing model-specific preparation tools only after genuine raw records and local assets are ready. Preparation keeps approvals disabled. Then complete real, evidence-backed human review and run:

```bash
python ci/run_model_family_audit.py --output family-readiness.json
```

This read-only audit uses the production training preflight. It does not download, approve, prepare, or train data. Training remains blocked until each model's provenance, integrity, governance, privacy, consent and technical checks pass. A successful preparation command is not legal approval or model release.

## Current boundary

This kit is only a collection aid. It does not make external data first-party, establish ownership, infer speaker consent, approve SAFE labels, enable training_allowed, or create checkpoints.
