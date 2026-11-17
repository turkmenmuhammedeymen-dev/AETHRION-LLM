# Project-owner authorization — cleared-data training

**Recorded:** 2026-11-17  
**Project:** AETHRION-LLM  
**Scope:** AETHRION-LLM repository and its designated Hugging Face Space only.

The project owner explicitly authorizes the team to review public dataset sources, prepare datasets, and train first-party AETHRION models **only after the exact dataset release and every included source item pass the project's license, provenance, privacy and consent gates**.

This authorization covers:
- Public metadata and source-rights review, evidence recording and source-specific allowlist creation.
- Non-destructive validation, hashing, deduplication, privacy filtering and train/validation/test preparation.
- From-random-initialization training for the eleven first-party model IDs after the model-specific manifest and data validator pass.
- Training checkpoints, resumable jobs, evaluation reports and release-readiness checks.
- Changes to this AETHRION-LLM repository and deployment to `vizyon2027/AETHRION-LLM` only.

This authorization does **not**:
- Grant copyright, database, privacy, publicity, speaker or other third-party rights.
- Replace per-item licenses, source-specific permissions, speaker consent, privacy review or provenance evidence.
- Authorize accepting gated provider click-through terms, signing contracts, sharing personal/contact details, or representing the user in a legal agreement.
- Authorize paid dataset purchases, paid compute, paid storage or hardware upgrades.
- Authorize using private user chats, personal data or synthetic test fixtures as production training data.
- Authorize modifying any other repository, including Vizyon2027 product repositories.

## Current gate decision

`permits_training` is true only for data that has already passed all source-specific checks. The global `training_approval_granted` and all dataset-level `training_allowed` flags remain false until each exact release and source item has documented rights for the intended ML-training use, required attribution, provenance, privacy/consent review, pinned revision, source and prepared-data SHA-256 hashes, deduplication and source-disjoint train/validation/test splits.

For speech data, the intended use and consent scope must explicitly cover the model objective. ASR permission does not automatically authorize TTS or voice synthesis. No voice cloning or speaker impersonation is authorized.

For mixed-source text, code, images and 3D assets, unknown or incompatible items must be excluded rather than inferred to be permitted. Commercial-intent product training must not use non-commercial-only sources.

## Audit trail

This file records the project owner's project-level instruction to proceed with cleared-data training. It is not a legal opinion, dataset clearance certificate or provider contract. Automated checks validate evidence fields; they cannot determine legal rights. A human must review the exact evidence before a model-specific gate is opened.

No dataset is approved by this authorization alone. No production training or production checkpoint is implied.
