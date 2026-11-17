# AETHRION full-family readiness audit

Run from the repository root:

```bash
python -m aethrion.family_audit
python -m aethrion.family_audit --output reports/family-readiness.json
```

The command audits all eleven IDs in `models/manifest.json`: chat, code,
vision, image, asr, tts, 3d, research, document, safety and vocoder. It checks
the expected architecture config, the canonical per-model data manifest,
explicit data approval flags, local artifact presence and registry status.

For specialist datasets, the canonical manifest path is
`data/specialists/manifests/<model_id>.json`. The existing
`data/specialists/manifest.template.json` is only a template and is never
treated as an approved dataset. BASE uses its corpus manifest and source
manifest.

This is an audit, not an approval or training command. It does not download
data, accept terms, alter any approval flag, create checkpoints, or mark a
model trained. Missing data manifests are reported as not ready. A release
requires the separate registry checks, real held-out metrics, safety review,
model card and human approval.
