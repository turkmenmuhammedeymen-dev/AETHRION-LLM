# AETHRION full-family readiness audit

Run from the repository root:

```bash
python -m aethrion.family_audit
python -m aethrion.family_audit --output reports/family-readiness.json
```

The command audits all eleven IDs in `models/manifest.json`: chat, code,
vision, image, asr, tts, 3d, research, document, safety and vocoder. It invokes
the production `train_family.preflight_model` separately for each model, so
training readiness is based on the same local-path, source-provenance, dataset
hash, tokenizer and human-approval checks that gate actual training. A pair
of true approval flags or a present manifest is not enough.

For specialist datasets, the canonical manifest path is
`data/specialists/manifests/<model_id>.json`. The existing
`data/specialists/manifest.template.json` is only a template and is never
treated as an approved dataset. BASE uses
`data/processed/corpus_manifest.json` and
`data/raw/cosmos/sources.json`.

Release readiness is checked through the native model registry, including
required local artifacts, SHA-256 integrity, held-out benchmark thresholds,
safety gates and human review evidence. Merely having model code or a
checkpoint file does not make a model releasable.

This is an audit, not an approval or training command. It does not download
data, accept terms, alter any approval flag, create checkpoints, or mark a
model trained. Missing data manifests and unverified sources are reported as
blocked. CI preserves a JSON readiness report as a workflow artifact for each
supported Python version.
