# Synthetic development fixtures

Run `python tools/build_synthetic_fixtures.py` to generate two JSONL fixtures for each of the 11 model IDs (22 records total).

These records are strictly for parser, schema, routing, and fail-closed pipeline tests. They are marked `ai_generated_synthetic`, `first_party_origin_attested: false`, `human_reviewed: false`, and `training_eligible: false`. The generated manifest always has `training_allowed: false`.

VISION, IMAGE, ASR, TTS, VOCODER, and 3D records contain metadata placeholders only, not actual media. SAFE labels are synthetic and unreviewed. Nothing in this directory constitutes licensed, owner-authored, human-reviewed production training data. Do not use it to unlock training.
