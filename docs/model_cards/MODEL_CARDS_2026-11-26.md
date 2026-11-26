# AETHRION model cards (draft) — 2026-11-26

All eleven models are **NOT_RELEASED**, experimental, first-party, small, trained on synthetic or Claude-drafted
data at the project owner's request. No dataset has passed a human rights/provenance/privacy review, and the
repository's approval gates are currently closed (audits now refuse to evaluate unapproved data — verified by a
failing audit run on 2026-11-26 with `Training blocked: specialist dataset and owner approvals are not ...`).
Figures below were measured earlier, on held-out group-split test sets, CPU training in GitHub Actions, while an
owner-statement attestation was in place. That attestation path has since been disabled (PR #164); the numbers
cannot be re-measured until a real human approval exists. They are history, not a release claim.

Common limits: synthetic data only; no real-world validation; no safety certification; do not deploy to users.

| Model | Purpose | Measured (held-out) | Known limits |
|---|---|---|---|
| CHAT | Turkish chat (BASE + instruction tune) | BASE test loss 4.12 (from 8.95); tuned held-out loss 0.040 | Not fluent. Tuned loss is suspiciously low: only 25% of fact groups were held out of BASE, so leakage is not excluded. Refusal examples were added but refusal behaviour was never generation-tested. |
| CODE | Code completion LM | next-token accuracy 0.937 vs majority baseline 0.921; test loss 0.467 | Marginal gain. 163 snippets; no executable pass-rate measured. |
| RESEARCH | Passage retrieval (dense + lexical overlap) | top-1 0.45, MRR 0.55 on 78 test records (random 0.013) | Synthetic templated passages; the lexical term does most of the work on unseen vocabulary. |
| SAFETY | ALLOW / REVIEW / BLOCK text classifier | accuracy 0.849, macro-F1 0.831 on 378 records (majority 0.444) | Roughly 1 in 7 decisions wrong; templated phrasing; unsuitable as a safety gate. |
| VISION | Image-text embedding similarity | no retained numbers | Not OCR, captioning or VQA. 360 synthetic geometric images. |
| IMAGE | Small text-to-image | no retained numbers | 64x64 flat-colour shapes only. |
| ASR | CTC speech recognition | no retained numbers (WER/CER never measured) | 160 synthetic tones/phrases; will not recognise real speech. |
| TTS | Text-to-spectrogram | no retained numbers | Synthetic voices only; intelligibility not evaluated. |
| VOCODER | Spectrogram-to-waveform | no retained numbers | Synthetic audio only. |
| 3D | Text-to-point-cloud | no retained numbers | 1024-point clouds of primitive shapes; not meshes. |
| DOCUMENT | Layout/tag extraction | structural accuracy ~0.999 on synthetic documents | Synthetic invoices/forms; no OCR; not a real document-understanding model. |

Binary assets for IMAGE/VISION/ASR/TTS/VOCODER/3D/DOCUMENT passed a full structural audit
(`docs/review/BINARY_ASSET_FULL_AUDIT_2026-11-26.md`); that is not a quality or rights review.
