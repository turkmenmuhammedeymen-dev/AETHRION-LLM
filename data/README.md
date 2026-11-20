# AETHRION training data

Keep raw/processed corpora, tokenizer files, and model checkpoints out of Git.
Use only owner-authored text, verified public-domain material, or datasets whose
license explicitly permits the intended ML training use.

1. Put UTF-8 .txt or .jsonl files (JSONL needs a string `text` field) under data/raw/.
2. Copy sources.example.json to sources.json and add one manifest entry per file.
3. Record origin, license, permitted use, and explicit training approval.
4. Run the preparation command from the repository root.

The pipeline rejects unlisted/unapproved files, normalizes Unicode and whitespace,
deduplicates records, groups records by provenance, and creates separate train,
validation and test files with SHA-256 audit metadata. At least three independent
source groups are required. The BASE tokenizer learns from the train-only split and
writes a provenance sidecar bound to the approved source manifest and split hashes.
These checks do not prove license validity, Turkish language quality, or absence of
personal/copyrighted material. Human review remains mandatory.


## Ten specialist models

The native family also has ten specialist trainers. Their fail-closed manifest
templates live in `specialists/manifests/`: code, vision, image, asr, tts,
vocoder, 3d, research, document and safety. The templates deliberately contain
empty source lists, no dataset hash, and false training/owner/privacy approvals.
They are intake scaffolding, not datasets. Do not change approval fields to
true merely to make preflight pass.

The empty base source inventory example is
`raw/cosmos/sources.example.json`. The base model still needs real owner-authored
source files, a reviewed source inventory, a prepared corpus, a train-only
tokenizer with provenance, and an audit report.

Required specialist NPZ arrays:

- code: `sample_id, group_id, split, token_ids`
- vision/image: `sample_id, group_id, split, images, token_ids`
- ASR: `sample_id, group_id, split, features, input_lengths, targets, target_lengths`
- TTS: `sample_id, group_id, split, token_ids, token_mask, token_lengths, frame_to_token, frame_mask, frame_lengths, target_mels, durations`
- vocoder: `sample_id, group_id, split, target_mels, target_waveforms, frame_lengths, waveform_mask`
- 3D: `sample_id, group_id, split, points, token_ids`
- research: `sample_id, group_id, split, query_ids, passage_ids`
- document: `sample_id, group_id, split, token_ids, boxes, layout_ids, tag_labels`
- safety: `sample_id, group_id, split, token_ids, decision_labels, reason_labels`

Audio datasets additionally require verified speaker consent. SAFE training
requires its own separately reviewed safety data. Synthetic tensors and unit
test fixtures must never be treated as production corpora. Group IDs and exact
duplicate model inputs must remain isolated across train, validation and test.
