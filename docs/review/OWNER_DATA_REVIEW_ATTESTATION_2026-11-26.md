# Project-owner review attestation — 26 November 2026

## Recorded owner declaration

On 26 November 2026, the project owner explicitly stated in the AETHRION maintenance conversation:

> All datasets have been reviewed and verified by the project owner, and the project owner authorizes the requested project work.

This document records that statement as an owner self-attestation for the AETHRION model families: CHAT/BASE, CODE, VISION, IMAGE, ASR, TTS, VOCODER, 3D, RESEARCH, DOCUMENT and SAFETY.

## Scope recorded

The owner states that they have reviewed and verified the dataset materials and authorize their use in AETHRION development and model training, subject to the technical and safety gates in the repository. This supersedes earlier repository notes that said the owner had not yet completed review.

For ASR, TTS and VOCODER, the current audio generator at `data/authored/make_synth_audio.py` documents procedural formant synthesis. It states that no human speaker is recorded or imitated. The source generator and provenance were inspected in this maintenance pass. The audio datasets are therefore recorded as synthetic-only; no real-speaker consent is represented as having been obtained. If real-speaker audio is added later, this attestation does not cover it and separate purpose-specific consent evidence is required.

## Important limits

- This is an owner self-attestation recorded from the owner's explicit statement. It is not a cryptographic signature and is not an independent legal opinion or third-party rights audit.
- Repository checks can validate file paths, hashes, manifest consistency, split integrity and technical preflight conditions; they cannot independently prove copyright ownership, privacy compliance, factual correctness or the identity of the person making the declaration.
- The approval applies only to the exact source and prepared-data revisions whose hashes are recorded in the corresponding manifests. Any data change requires new hashes and renewed review.
- Training approval is not a model-quality certification or permission to publish checkpoints. Evaluation, safety testing, artifact integrity and release controls remain separate.
- No model training is initiated by this document or by the pull request that adds it.

## Recorded reviewer and timestamp

- Reviewer role: project owner, self-attestation as explicitly declared in the maintenance conversation.
- Recorded at: 2026-11-26T18:06:04.915Z.
- Evidence source: project-owner declaration in the AETHRION maintenance conversation on 26 November 2026.
