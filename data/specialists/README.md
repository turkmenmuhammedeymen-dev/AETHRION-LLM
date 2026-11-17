# AETHRION specialist dataset format

All specialist datasets are prepared locally as a NumPy NPZ archive loaded with
`allow_pickle=False`. The archive must contain `sample_id` and `split`
arrays. Every record must be assigned exactly one of `train`, `validation`
or `test`; all three splits must be non-empty and sample IDs must be unique.

| Model | Required NPZ arrays |
|---|---|
| BASE instruction tuning | `sample_id, split, input_ids, labels` |
| Code | `sample_id, split, token_ids` |
| Vision | `sample_id, split, images, token_ids` |
| Image | `sample_id, split, images, token_ids` |
| ASR | `sample_id, split, features, input_lengths, targets, target_lengths` |
| TTS | `sample_id, split, token_ids, frame_to_token, target_mels, durations` |
| Vocoder | `sample_id, split, target_mels, target_waveforms` |
| 3D | `sample_id, split, points, token_ids` |
| Research | `sample_id, split, query_ids, passage_ids` |
| Document | `sample_id, split, token_ids, boxes, layout_ids, tag_labels` |
| Safety | `sample_id, split, token_ids, decision_labels, reason_labels` |

The manifest is fail-closed. It requires exact boolean approval fields,
reviewed sources with local source paths and matching SHA-256 values, license
and training permission evidence, privacy review, and speaker consent review
for ASR/TTS/Vocoder. Do not set an approval to true unless the responsible human has
actually reviewed and documented that evidence.

Each source entry must contain:
`source_id`, `source_uri`, `source_path`, `sha256`, `license`,
`permitted_use`, `evidence_uri`, `allow_training: true` and
`privacy_reviewed: true`. ASR/TTS/Vocoder also require `speaker_consent_verified: true`. Paths are relative to the manifest folder.


For AETHRION-BASE instruction tuning, use `model_id: "chat"` and
`modality: "text"`. The `input_ids` array contains the prompt and response;
`labels` must be `-100` at prompt positions and contain target token IDs
only at positions the model is expected to learn. The manifest must include
`tokenizer_sha256` matching the tokenizer and verified base checkpoint.
Do not include private chats or personal messages without documented consent
and legal permission.

SAFE manifests must declare `safety_min_per_class` (minimum 25) and provide at least that many ALLOW, REVIEW and BLOCK examples in each of train, validation and test. `reason_labels` are binary multi-hot vectors. This is only a minimum data-quality gate; it does not establish classifier accuracy, calibration, robustness or policy coverage. Do not manufacture harmful operational examples; use reviewed taxonomy labels and benign, prevention-oriented examples.

The TTS model learns transcript-to-mel alignment and durations. The separately
trained Vocoder learns mel-to-waveform conversion; both datasets require
documented speaker consent. TTS cannot be released unless its companion
AETHRION-VOCODER also passes its own evaluation and release gates.

## Training

Create a per-model manifest by copying `manifest.template.json`, filling the
dataset identity, modality, local NPZ path/hash and reviewed source evidence.
Keep it unapproved until all legal, privacy and consent checks are complete.

Run one specialist, using its matching config:

```bash
python -m aethrion.specialist_train \
  --config configs/specialists/code.json \
  --manifest data/specialists/code/manifest.json
```

Resume only from the same model and same dataset hash:

```bash
python -m aethrion.specialist_train \
  --config configs/specialists/code.json \
  --manifest data/specialists/code/manifest.json \
  --resume artifacts/models/code/checkpoint_last.pt
```

Training produces a checkpoint, SHA-256 sidecar, metrics and a
`training_report.json` marked `NOT_RELEASED`. It does not change the model
registry to `trained`. Independent held-out benchmarks, safety review,
processor/tokenizer integrity and explicit release approval remain mandatory.

These files define the training data contract. No user dataset has been
approved or trained by adding this template.
