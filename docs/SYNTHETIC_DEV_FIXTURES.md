# Synthetic development fixture bootstrap

Run from the repository root:

```bash
python scripts/generate_dev_fixtures.py
python -m unittest tests.test_dev_fixture_generator -v
```

The generator creates a tiny deterministic fixture tree under
`data/dev_fixtures/`. It is intentionally outside `data/raw` and the
production specialist manifest paths.

## Scope

- BASE, CHAT, CODE, RESEARCH, DOCUMENT and SAFE receive tiny UTF-8 text
  examples. SAFE labels are deliberately absent and marked as unreviewed.
- VISION and IMAGE receive abstract PPM rasters for image-decoder smoke tests.
- 3D receives random normalized NPY point arrays for point-cloud reader tests.
- ASR, TTS and VOCODER receive no fabricated audio. Tones or noise are not
  speech and would create misleading transcript/alignment labels.

## Hard boundary

The fixture manifest sets `training_allowed`, `owner_approval`,
`privacy_reviewed`, and `human_reviewed` to false. Every model is marked
`production_training_ready: false`. The fixtures are not owner-authored
production data, not human-reviewed labels, not a useful corpus, and not
benchmark or release evidence. Do not copy them into `data/raw` or
`data/specialists/manifests`, and do not pass them to a training command.

This bootstrap is for checking parsers and file contracts only. Actual model
training still requires sufficiently large, diverse, reviewed, rights-cleared
data for each modality. Audio requires explicit speaker permission and real
reviewed recordings; SAFE labels require independent human review. The existing
family trainer and model audit remain the authoritative fail-closed gates.
