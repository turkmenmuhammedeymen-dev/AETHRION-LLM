# CHAT pilot data — provenance (OWNER-ATTESTED 2026-11-21, see GOVERNANCE.json; records changed after the sha256 in that file need re-attestation)

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
