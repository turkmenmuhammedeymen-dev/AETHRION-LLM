# Synthetic family fixtures (not production training data)

Generate deterministic, local-only software fixtures for all 11 AETHRION model
pipelines:

```bash
pip install -e ".[visual]"
python tools/generate_synthetic_family_fixtures.py --output data/synthetic/fixtures
python -m unittest tests.test_synthetic_family_fixtures -v
```

The generator creates:
- CHAT, CODE, RESEARCH, DOCUMENT and SAFE JSONL inputs;
- VISION and IMAGE geometric PNG assets and JSONL metadata;
- 3D normalized random point-cloud NPY assets and JSONL metadata;
- ASR, TTS and VOCODER 16 kHz non-speech WAV tone fixtures and JSONL metadata;
- a family manifest recording counts, seed and explicit non-production status.

The integration test runs all eleven fixture families through the repository's
native text, visual and audio preparation adapters. CHAT uses a temporary
byte-BPE tokenizer artifact to exercise its tokenizer-reuse contract. The
resulting specialist manifests remain unapproved and cannot pass training
preflight.

## Non-negotiable boundary

Every record is tagged `synthetic_fixture=true` and
`production_training_approved=false`. The family manifest sets
`not_for_training=true`, `training_allowed=false`, and all review/consent
flags to false. The generated audio contains tones, not speech or speaker
recordings. Images and point clouds are simple synthetic geometry.

These assets are useful for parser, file-path, tensor-shape, feature extraction,
tokenizer, provenance and integration tests only. They are not representative
capability data and must never be used to train or claim readiness for a
production model. Do not edit the manifest to make these fixtures eligible for
training.

Production data still requires separately documented first-party authorship,
rights-holder authority, permitted training use, privacy/quality review,
group-disjoint splits, and (for audio) valid speaker consent. SAFE additionally
requires independent human review of every safety label and held-out evaluation.
No generated fixture can satisfy those requirements.

The generator uses no external dataset and explicitly excludes Common Voice,
ShapeNet, CodeSearchNet and LibriSpeech.
