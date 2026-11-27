# Specialist checkpoint evaluation — 26 November 2026

## Scope and decision

This report records GitHub Actions audit run [45837980011](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/actions/runs/45837980011), which produced artifact `aethrion-specialist-evaluation`.

**Release decision: 0/10 specialist checkpoints released.** The audit explicitly returned an empty `released_models` list. This report does not approve data, training, or release.

The run evaluated private checkpoints against the then-current prepared specialist manifests. The prepared data are synthetic pilot candidates. Evaluation on a small synthetic held-out split is not evidence of real-world task quality.

## Results

| Family | Checkpoint step / configured max | Integrity | Held-out samples | Result |
|---|---:|---|---:|---|
| CODE | Not evaluated | FAIL | — | Checkpoint dataset hash does not match current manifest; retraining from the corrected dataset is required after the approval gate passes |
| RESEARCH | 799 / 800 | PASS | 78 | No evidence of learning: held-out loss 1.849 vs untrained baseline 1.347 |
| SAFETY | 1499 / 1500 | PASS | 378 | Accuracy 0.849; macro-F1 0.831; synthetic-only; below the proposed release thresholds |
| VISION | 1999 / 2000 | PASS | 28 | Synthetic pilot measurement only; no real-image task validation |
| IMAGE | 1999 / 2000 | PASS | 19 | Synthetic pilot measurement only; no independent image-quality/reconstruction evaluation |
| ASR | 1999 / 2000 | PASS | 11 | Synthetic audio only; no human-speech intelligibility evaluation |
| TTS | 1999 / 2000 | PASS | 11 | Synthetic audio only; no human listening or naturalness evaluation |
| VOCODER | 1999 / 2000 | PASS | 11 | Synthetic audio only; no human listening or waveform-quality evaluation |
| 3D | 1999 / 2000 | PASS | 22 | Synthetic point-cloud pilot only; no mesh/geometry quality validation |
| DOCUMENT | 1999 / 2000 | PASS | 36 | Synthetic layout pilot only; no independent document/OCR benchmark |

For the nine integrity-passing checkpoints, the artifact reported finite weights and matching checkpoint/dataset hashes. It also reported `training_report: null` for every family, so the audit artifact alone does not provide the complete training provenance record.

## Governance state at audit time

All ten specialist manifests on `main` remain fail-closed:
- `training_allowed=false`
- `owner_approval=false`
- `privacy_reviewed=false`
- Audio families additionally require speaker-consent evidence.

The project-owner file found on branch `chore/record-owner-data-review-approval` is `docs/OWNER_DATA_REVIEW_AUTHORIZATION_2026-11-17.md`. It authorizes source review and non-destructive preparation planning only. It explicitly does not approve data, grant third-party rights, or open training gates. That branch is 214 commits behind current `main` and is not suitable for direct merge.

## CHAT / BASE

The repository contains 93 authored BASE JSONL source files and `data/authored/chat/records.jsonl`, but no approved prepared BASE corpus. `data/processed/corpus_manifest.json` remains `training_allowed=false` and `first_party_origin_attested=false`. The deprecated BASE preparer intentionally refuses to manufacture approvals. A candidate corpus may be assembled for review, but it must not be treated as an approved training corpus.

## Required next steps

1. Record an up-to-date, explicit owner declaration that accurately identifies which files/records are claimed first-party project data and their actual creation method (including Claude-generated content); do not describe generated records as personally hand-authored.
2. Complete source/rights, privacy, factual-quality and safety review for the exact snapshots. For ASR/TTS/VOCODER, document the synthetic signal-generation method and verify that no real speaker recordings or imitations are present; if any real speaker data exist, obtain purpose-specific consent evidence.
3. Build and hash a reviewable BASE candidate corpus from the existing 93 source files and CHAT records; keep its manifest blocked until review passes.
4. After approvals are evidence-backed, rebuild CODE preparation, retrain CODE from scratch against the corrected dataset hash, and rerun the audit.
5. Run task-appropriate external/independent evaluations for every family; do not release any model solely because a synthetic loss improved.

## Limitations

The audit used synthetic pilot test splits and very small support counts for several modalities. It does not validate model usefulness, safety in deployment, legal rights, or release readiness. No model was published by this run.
