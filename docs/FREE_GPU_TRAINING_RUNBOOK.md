# AETHRION training compute status

This document records the current approved compute boundary. It intentionally
does not provide Kaggle, Colab, Codespaces or other external training setup
instructions because those environments are outside the current project scope.

## Current compute status

- **GitHub Actions:** approved for automated tests, CPU smoke tests and
  readiness reports. It is not a persistent GPU training worker.
- **Hugging Face Space ZeroGPU:** approved for short request-scoped inference
  and architecture preflight checks. It is not a long-running model-training
  environment.
- **Kaggle, Google Colab and Codespaces:** not in the current authorized scope.
  Do not create notebooks, sessions, tokens or compute jobs there.
- **Paid hardware, paid accelerators, paid datasets and paid storage:** not
  authorized.

## Before real training can begin

Two independent prerequisites remain unresolved:

1. **Data:** create genuine first-party owner-authored material for the exact
   model, document authorship and rights, complete privacy/quality review,
   deduplicate it, create group-disjoint train/validation/test splits, and
   obtain the required human approval. Audio models additionally require
   speaker consent for the exact objective. SAFE requires independent human
   review of its labels.
2. **Compute:** explicitly authorize and configure a suitable training
   environment that is within the project scope. A short ZeroGPU request or a
   passing CPU test is not a substitute for a persistent training worker.

Until both prerequisites are met, keep every training gate closed. Do not
create synthetic production corpora, change approval flags to true, or mark
any model trained.

## Safe checks currently available

Run the repository's automated test suite through GitHub Actions. Use the
Hugging Face Space's **11-model architecture preflight** only to check model
construction, device allocation, forward/backward passes and finite losses
with disposable synthetic tensors. It does not run optimizer steps, save
checkpoints, update model statuses or train any model.

## Release requirements

A successful training job is not a release. Each model still requires a
verified checkpoint, source-bound training report, held-out evaluation,
model-specific quality benchmarks, safety review, model card, artifact
integrity checks, and explicit human release approval. The native registry
must continue to report unavailable until all requirements pass.
