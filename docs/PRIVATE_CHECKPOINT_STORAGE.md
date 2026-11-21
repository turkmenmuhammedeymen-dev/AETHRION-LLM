# Private checkpoint storage for Hugging Face Space

A ZeroGPU request is temporary. Local files in the request container must not be
assumed to survive the next request. AETHRION's checkpoint artifact adapter uses
an existing **private Hugging Face Dataset repository** as a separate artifact
store; it does not store model weights in the public Space source repository.

## Required Space Secrets

Configure both values in the Space Settings → Variables and secrets:

- `HF_TOKEN`: a write-scoped token for the private checkpoint Dataset repository.
- `AETHRION_CHECKPOINT_REPO`: the existing `owner/repository` ID of that Dataset.

The adapter refuses missing credentials, inaccessible repositories, public
repositories, unsafe model IDs, invalid pointers, and checksum mismatches. It
never creates a repository, changes visibility, prints a token, or silently
falls back to local-only storage. Create and verify the private Dataset
repository and its access policy before configuring these secrets.

## Generation protocol

Each checkpoint is uploaded under a SHA-256-addressed immutable filename.
Its checksum sidecar is uploaded next. The small `latest.json` pointer is
published last, so a partial upload is not advertised as the latest generation.
Restore validates the pointer, downloads the immutable checkpoint and sidecar,
checks SHA-256, and atomically places the verified file at the requested local
path.

Only one training writer per model is supported at a time. Do not run concurrent
training jobs against the same model pointer. Keep dataset manifests and
training reports alongside the run record; this adapter is specifically for
checkpoint bytes and their integrity metadata.

## Current boundary

This adapter is a storage primitive, not a training approval mechanism or a
Space training UI. It does not open any data gate, start training, create model
weights, or mark a model trained. The app must explicitly call save/restore
around a bounded training session before cross-request resume is enabled.
