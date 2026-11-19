# Synthetic BASE pilot dataset

This generator creates a small, original Turkish synthetic corpus for validating
AETHRION's data-preparation pipeline. It is **not** a useful foundation-model
corpus, is not owner-authored material, and does not grant training approval.

## Generate locally

```bash
python scripts/build_synthetic_base_pilot.py
python -m aethrion.prepare_data \
  --input data/raw/synthetic_base_pilot \
  --manifest data/raw/synthetic_base_pilot/sources.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --output data/processed/synthetic-base-pilot.txt \
  --report data/processed/synthetic-base-pilot-report.json \
  --min-records 100
```

The generated directory is local under `data/raw` and should remain uncommitted.
The manifest labels every source as `ai_generated_synthetic` and sets
`allow_training=false`. The dataset-level gate remains closed.

## Before any training

A responsible adult reviewer must inspect the generated content, quality,
provenance and intended use. The project owner must explicitly adopt a synthetic
data policy and the governance validator must be updated to represent synthetic
origin honestly rather than falsely claiming first-party human authorship.
Only after that review may an authorized maintainer create a new, accurately
scoped approval record. Do not change approval flags merely to make the trainer
run. A small synthetic smoke corpus is suitable for pipeline checks, not for
claiming a capable conversational model.

The BASE trainer additionally requires a trusted HMAC audit key, a signed
corpus report, a tokenizer trained only on the approved training split, and
matching provenance hashes. Never put the HMAC key or raw training artifacts
in Git.
