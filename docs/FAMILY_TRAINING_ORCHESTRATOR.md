# AETHRION family training orchestrator

The repository now has separate BASE and specialist trainers. This command
provides one deterministic queue across all eleven model IDs without merging
their weights or their data.

## Preflight only (default)

```bash
python -m aethrion.train_family
python -m aethrion.train_family --report reports/family-training-preflight.json
```

The command validates the BASE corpus/source/corpus-report/tokenizer chain and
the exact NPZ manifest for each specialist. It does not initialize models,
optimizers, or GPUs in preflight mode.

## Execute approved models

```bash
python -m aethrion.train_family --execute
python -m aethrion.train_family --models safety asr --execute
```

Execution is sequential. Each model is independently preflighted and only
then passed to its own trainer. A failed or blocked model does not authorize
another model's data. Existing checkpoints are resumed only from the
model-specific output path; each trainer verifies the checkpoint and dataset
identity. The orchestrator never changes `models/manifest.json` to mark a
model trained or released.

BASE uses `aethrion.train` and its corpus/source approval chain. CODE, VISION,
IMAGE, ASR, TTS, 3D, RESEARCH, DOCUMENT, SAFE and VOCODER use
`aethrion.specialist_train` and
`data/specialists/manifests/<model_id>.json`.

A successful trainer process remains `NOT_RELEASED`. Independent held-out
benchmarks, model-specific safety and quality gates, signed artifact hashes,
model cards, and human release evidence are still required. The command
contains no third-party model weights or inference fallback.

Do not set approval flags to true to make the queue run. They must reflect
actual source-rights, provenance, privacy and (for audio) speaker-consent
review. Synthetic fixtures are software tests only and are never production
training data.
