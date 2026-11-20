# All-model synthetic fixture pipeline

Generate disposable test inputs for the complete native model family:

```bash
python scripts/build_all_model_synthetic_fixtures.py --output data/synthetic_fixtures
python -m pytest -q tests/test_all_model_synthetic_fixtures.py
```

The generator creates four deterministic records per model and tiny local assets
where required: PPM patterns for VISION/IMAGE, generated WAV tones for
ASR/TTS/VOCODER, and JSON point sets for 3D. CHAT, CODE, RESEARCH, DOCUMENT
and SAFE receive schema-shaped JSONL records. These are intentionally tiny
software fixtures, not useful foundation-model training corpora.

Every generated manifest and model entry is marked `ai_generated_synthetic`,
`training_allowed: false`, and not reviewed. The generator does not read,
modify, or approve `data/processed/corpus_manifest.json`,
`data/specialists/manifests/*.json`, or `models/manifest.json`. It does not
create tokenizers, checkpoints, training reports, or release artifacts.

Generated output belongs in the ignored `data/synthetic_fixtures/` directory
and must not be committed or uploaded as a production dataset. The fixtures
validate deterministic file creation, UTF-8, modality asset handling, hashes,
and manifest labeling only. They do not prove model quality, dataset rights,
speaker consent, privacy review, or training readiness.

Production training remains blocked until each model's actual corpus has
documented provenance and rights, required human review, privacy/quality
checks, group-disjoint splits, and any modality-specific consent. Audio models
require explicit speaker consent; SAFE requires a separate human safety review.
A responsible adult should support legal, privacy, and consent review.
