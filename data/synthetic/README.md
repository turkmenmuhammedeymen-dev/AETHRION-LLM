# Synthetic family fixtures (not production training data)

Generate deterministic, local-only software fixtures for all 11 AETHRION model
pipelines:

```bash
pip install -e '.[visual]'
python tools/generate_synthetic_family_fixtures.py --output data/synthetic/fixtures
```

The generator creates:
- CHAT, CODE, RESEARCH, DOCUMENT and SAFE JSONL inputs;
- VISION and IMAGE geometric PNG assets and JSONL metadata;
- 3D normalized random point-cloud NPY assets and JSONL metadata;
- ASR, TTS and VOCODER 16 kHz non-speech WAV tone fixtures and JSONL metadata;
- a family manifest that records counts, seed and explicit non-production status.

SAFE emits 225 placeholder-labelled records so the current preparer's per-split
class-count shape can be exercised. These labels are **not** safety annotations.

## Non-negotiable boundary

Every record is marked `synthetic_fixture=true` and
`production_training_approved=false`. The family manifest sets
`not_for_training=true`, `training_allowed=false`, and every review/consent
flag to false. The generated audio contains tones, not speech or speaker
recordings. The images and point clouds are simple synthetic geometry.

These assets are useful for parser, file-path, tensor-shape, provenance and
integration tests only. They are not representative capability data and must
never be used to train or claim readiness for a production model. Do not edit
the manifest to make these fixtures eligible for training.

Production data still requires separately documented first-party authorship,
rights-holder authority, permitted training use, privacy/quality review,
group-disjoint splits, and (for audio) valid speaker consent. SAFE additionally
requires independent human review of every safety label and held-out evaluation.
No generated fixture can satisfy those requirements.

The generator uses no external dataset and explicitly excludes Common Voice,
ShapeNet, CodeSearchNet and LibriSpeech.
