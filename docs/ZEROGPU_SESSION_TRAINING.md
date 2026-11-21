# AETHRION-LLM ZeroGPU resumable training sessions

This document describes bounded training invocations for the native AETHRION
trainers. It does not grant dataset approval, create persistent storage, or
release a model.

## Session interface

All three entry points accept `--session-steps N`:

```bash
python -m aethrion.train --config configs/small.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --source-manifest data/raw/cosmos/sources.json \
  --source-root data/raw/cosmos \
  --corpus-report data/processed/corpus-report.json \
  --session-steps 20
```

For a specialist:

```bash
python -m aethrion.specialist_train \
  --config configs/specialists/code.json \
  --manifest data/specialists/manifests/code.json \
  --session-steps 20
```

For a selected family member through the orchestrator:

```bash
python -m aethrion.train_family --execute --models chat --session-steps 20
```

A session limit counts optimizer steps completed during that invocation. If
the configured training horizon is not complete, the trainer writes and
verifies its normal checkpoint, records an interrupted/not-released status,
and exits. Resume using the same configuration, data, tokenizer and approval
manifest, and the existing model-specific checkpoint. The orchestrator does
not automatically mark a model trained or released.

## Important ZeroGPU and storage boundary

ZeroGPU provides request-scoped GPU execution, not an unlimited persistent
worker. The session-step limit is a cooperative bound on optimizer steps; it
is not a hard wall-clock deadline. A single optimizer step, data loading,
evaluation, or checkpoint write can still exceed a request's remaining time.

The checkpoint code writes atomically and verifies SHA-256 locally. That
protects files while they remain available, but **does not itself make the
Space filesystem persistent**. Before relying on cross-request resume, verify
that the Space has persistent storage mounted at the configured artifact
location, or provide a separately secured private artifact store. Never put
raw private datasets, credentials, or checkpoints in a public Git repository.
Do not silently fall back to a public bucket or third-party model service.

Do not launch all eleven models in one request. Train one model at a time,
persist its checkpoint and sidecar, verify both after the next request, and
then resume. BASE and every specialist retain independent datasets,
tokenizers/processors, checkpoints and evaluation reports.

## Release gate

A successful session means only that the trainer safely paused or completed
its configured optimization horizon. It does not establish model quality.
Each model still requires held-out evaluation, integrity verification,
model-specific safety review, a model card, and explicit human release
approval. Dataset/source rights, privacy, provenance, quality and applicable
speaker-consent gates remain mandatory before the first optimizer step.


## Space readiness panel

The Hugging Face Space includes a **public, read-only 11-model training readiness**
panel. It calls the family orchestrator with `execute=False` and displays the
actual preflight result and blocker reason for each model. The report contains
only repository-level gate status; it does not expose private datasets,
credentials or model artifacts. The panel cannot invoke an optimizer, write a
checkpoint, alter a manifest, or change an approval flag. A READY result means
only that preflight passed; it is not a trained or released model.

Inference remains separately protected by Space authentication. The Space smoke
test verifies that training readiness works without inference credentials,
contains eleven model rows, and explicitly reports that training did not start
and no checkpoint was written.

## Private artifact sync and session wrapper

`aethrion/artifact_sync.py` mirrors `artifacts/models/<model>/` to a **private** Hugging Face model repo
(`AETHRION_ARTIFACT_REPO`, secret `HF_TOKEN`); it refuses public repos and non-allow-listed file names.
`aethrion/space_session.run_session(model, steps)` does pull -> gate-checked `train_family --execute` -> push for
one model and never marks anything released. Setup (owner): create the private repo, add `HF_TOKEN` and
`AETHRION_ARTIFACT_REPO` as Space secrets. Training still fails closed until the data gate is opened by the owner.
