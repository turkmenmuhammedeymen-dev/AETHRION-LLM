# SAFETY data — provenance (NOT YET ATTESTED; see docs/review/SAFETY_REVIEW.md)
1894 short, original Turkish requests labelled ALLOW / REVIEW / BLOCK, written for AETHRION-LLM on 2026-11-22 at the
owner's request (core set by hand + combinatorial templates in `make_safety.py`). Not copied from any dataset, site or person.
BLOCK texts are abstract statements of intent only — no operational instructions. Reason IDs: 0 benign, 1 physical harm,
2 self-harm, 3 cyber abuse, 4 fraud, 5 privacy abuse, 6 dangerous substances, 7 exploitation, 8 prompt injection,
9 ambiguous high-impact. Templated data has LOW diversity: a model trained on it is a pilot gate, never a production barrier
(see docs/AETHRION_SAFE_POLICY_V1.md). Separate human safety review is required (`safety_reviewed`).


## Human review status: PENDING
The project owner has confirmed that these records have not yet been individually reviewed. Claude-generated content and programmatically prepared assets remain subject to record-level checks for factual quality, privacy, source rights, duplication and modality-specific risks. This provenance note is not an approval to train.


## Updated owner review status — 26 November 2026

The project owner explicitly states that all dataset materials in the AETHRION model families have been reviewed and verified, and authorizes their use for project development and model training subject to technical and safety gates. This owner self-attestation supersedes earlier pending-review statements in this file. See [the recorded owner attestation](../../docs/review/OWNER_DATA_REVIEW_ATTESTATION_2026-11-26.md). This is not an independent legal or third-party rights audit; exact source and prepared-data hashes and technical preflight remain mandatory.
