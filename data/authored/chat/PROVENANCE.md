# CHAT pilot data — provenance (NOT YET HUMAN-REVIEWED 2026-11-21, see GOVERNANCE.json; records changed after the sha256 in that file need re-attestation)

- Content: original Turkish prompt/response pairs written for AETHRION-LLM (not copied from books, sites,
  datasets, repositories or private chats). Authored in a Claude Code session on 2026-11-21 at the project owner's request.
- No personal data, secrets, URLs or contact details (checked by `scripts/validate_authored_chat.py`).
- Size: pilot scale only (grows by rounds; see records.jsonl). Too small to produce a useful chat model; it exercises the pipeline.
- Groups: `group_id` = topic + conversation block; each group must stay inside one split.

## Owner attestation (to be completed by the responsible human — NOT done by the assistant)
- [ ] I reviewed every record for quality and accuracy.
- [ ] I confirm the project owns these texts and permits their use to train AETHRION models.
- [ ] I accept responsibility for rights/privacy review.

`training_allowed` / owner approval flags must stay false until the above is checked and recorded by the owner.


## Human review status: PENDING
The project owner has confirmed that these records have not yet been individually reviewed. Claude-generated content and programmatically prepared assets remain subject to record-level checks for factual quality, privacy, source rights, duplication and modality-specific risks. This provenance note is not an approval to train.


## Updated owner review status — 26 November 2026

The project owner explicitly states that all dataset materials in the AETHRION model families have been reviewed and verified, and authorizes their use for project development and model training subject to technical and safety gates. This owner self-attestation supersedes earlier pending-review statements in this file. See [the recorded owner attestation](../../docs/review/OWNER_DATA_REVIEW_ATTESTATION_2026-11-26.md). This is not an independent legal or third-party rights audit; exact source and prepared-data hashes and technical preflight remain mandatory.
