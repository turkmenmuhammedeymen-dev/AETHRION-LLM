# Project-owner authorization — dataset review

**Recorded:** 2026-11-17  
**Project:** AETHRION-LLM  
**Scope:** Dataset-source review and non-destructive preparation planning

The project owner explicitly authorized proceeding with the AETHRION-LLM dataset review and preparation-planning work in the conversation on 2026-11-17.

This authorization covers:
- Reviewing public dataset cards, source terms, release metadata and provenance documentation.
- Recording source-specific blockers and evidence URLs.
- Designing and testing non-destructive validation, hashing, deduplication and split-checking workflows.
- Preparing manifests and bounded preflight tooling that do not download restricted data or start training.

This authorization does **not**:
- Grant copyright, database, privacy, publicity, speaker, or other rights owned by dataset contributors or third parties.
- Replace a dataset provider's click-through terms, access approval, consent requirements, or per-item license review.
- Authorize accepting legal agreements or sharing the user's personal/contact information with dataset maintainers.
- Approve any dataset or source item for model training.
- Change the global or specialist training gates.
- Authorize paid compute, paid storage, or long-running jobs.

## Current gate decision

The fields `training_approval_granted` and `training_allowed` remain `false` until each exact dataset release and source item has documented rights for the intended ML-training use, required attribution, provenance, privacy/consent review, pinned revision, source and prepared-data SHA-256 hashes, deduplication, and source-disjoint train/validation/test splits.

For speech data, the intended use and consent scope must be verified separately. Dataset availability or a CC0 label alone is not a substitute for checking the release metadata and contributor assurances. No voice cloning or speaker impersonation is authorized.

For mixed-source text, code, images and 3D assets, unknown or incompatible items must be excluded rather than inferred to be permitted. COCO image rights, Objaverse per-asset licenses, Cosmos underlying source rights, and The Stack per-repository/per-file terms remain separate review items.

## Audit trail

This file records the project owner's instruction to continue the review. It is not a legal opinion or a dataset clearance certificate. Automated tests validate evidence fields; they cannot determine legal rights. A human must review the exact evidence before a model-specific gate is opened.

No dataset was downloaded, no training was started, and no production checkpoint was created by recording this authorization.
