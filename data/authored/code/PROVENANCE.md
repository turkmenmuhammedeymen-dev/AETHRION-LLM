# CODE pilot data — provenance (NOT YET HUMAN-REVIEWED 2026-11-21, see GOVERNANCE.json; records changed after the sha256 in that file need re-attestation)
The initial generator contains 27 seed programs (Python, JavaScript, SQL, Bash, HTML, CSS). The current `records.jsonl` contains 163 rows across 84 groups, expanded by multiple generators. The seed and expansion templates were generated at the project owner's request and have not been individually human-reviewed. Regenerate only after reviewing generator inputs with `python3 data/authored/code/make_code.py`;
validate with `python3 scripts/validate_authored_chat.py data/authored/code/records.jsonl id,group_id,text`.
Pilot scale only. Owner review/attestation is required before any approval flag is set (same checklist as CHAT).


## Human review status: PENDING
The project owner has confirmed that these records have not yet been individually reviewed. Claude-generated content and programmatically prepared assets remain subject to record-level checks for factual quality, privacy, source rights, duplication and modality-specific risks. This provenance note is not an approval to train.


## Updated owner review status — 26 November 2026

The project owner explicitly states that all dataset materials in the AETHRION model families have been reviewed and verified, and authorizes their use for project development and model training subject to technical and safety gates. This owner self-attestation supersedes earlier pending-review statements in this file. See [the recorded owner attestation](../../docs/review/OWNER_DATA_REVIEW_ATTESTATION_2026-11-26.md). This is not an independent legal or third-party rights audit; exact source and prepared-data hashes and technical preflight remain mandatory.
