# AETHRION first-party data authoring guide

This folder defines the owner-authored record formats for all 11 native model families. The templates are intentionally blank: they are not training examples and do not approve any data.

## Model record formats

| Model | Required record fields |
|---|---|
| BASE | `.txt` lines or JSONL `text`; optional `group_id` |
| CHAT | `id`, `group_id`, `prompt`, `response` |
| CODE | `id`, `group_id`, `text` |
| VISION | `id`, `group_id`, `image_path`, `caption` |
| IMAGE | `id`, `group_id`, `image_path`, `caption` |
| ASR | `id`, `speaker_id` or `group_id`, `wav_path`, `transcript` |
| TTS | `id`, `speaker_id` or `group_id`, `wav_path`, `transcript`, `durations` |
| VOCODER | `id`, `speaker_id` or `group_id`, `wav_path` |
| 3D | `id`, `group_id`, `point_path`, `text` |
| RESEARCH | `id`, `group_id`, `query`, `passage` |
| DOCUMENT | `id`, `group_id`, `text`, `boxes`, `layout_ids`, `tag_labels` |
| SAFE | `id`, `group_id`, `text`, `decision`, `reasons` |

## Authoring rules

1. Write or create each record from first-party material whose authorship and training rights can be documented. Do not copy books, websites, repositories, datasets, or other protected content without a verified training license.
2. Do not put private chats, passwords, API keys, personal records, addresses, contact details, or other identifying data into a corpus.
3. Use stable `group_id` values. All records from the same author, speaker, source document, conversation, or underlying work must remain in one split. The preparation pipeline hashes group identifiers before storing them in NPZ files.
4. Keep source files immutable after review. If any source changes, regenerate hashes and repeat review.
5. Review content quality and labels by a human. SAFE labels require a separate safety review and balanced ALLOW/REVIEW/BLOCK coverage.
6. Audio requires explicit permission from each speaker for recording and AI training. Do not infer consent from possession of a file.
7. Images require rights and privacy review, including identifiable people, private locations, and embedded metadata. 3D assets require rights review for both geometry and text descriptions.
8. Create the pending human-review template with `python -m aethrion.data_governance --model MODEL_ID --output REVIEW.json`. Complete it only after reviewing the actual evidence.
9. Keep `training_allowed`, `owner_approval`, source `allow_training`, privacy and consent flags false until the human review is complete. The validator will reject incomplete or mismatched approvals.

## Preparation commands

Use the existing model-specific preparation modules after records and local assets are ready:

- BASE: `python -m aethrion.prepare_data --help`
- CHAT/CODE/RESEARCH/DOCUMENT/SAFE: `python -m aethrion.specialist_prepare --help`
- ASR/TTS/VOCODER: `python -m aethrion.specialist_prepare_audio --help`
- VISION/IMAGE/3D: `python -m aethrion.specialist_prepare_visual --help`

Preparation can run while approvals are pending. It creates normalized, hashed, group-disjoint datasets but does not authorize training. The independent training approval and release gates remain mandatory.

## Reality check

The templates are a workflow scaffold, not a corpus. A few records or automatically generated synthetic examples are not enough to train useful foundation models. Each model needs a sufficiently large, diverse, human-reviewed dataset and held-out evaluation data. This repository must not mark any model trained until real training, evaluation, and release evidence exists.
