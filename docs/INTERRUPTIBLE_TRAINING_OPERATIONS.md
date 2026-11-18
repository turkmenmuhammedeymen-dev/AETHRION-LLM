# AETHRION interruptible training operations

This runbook describes the repository-local training session journal and the
existing per-model checkpoint/resume contract. It does not grant data approval
or start training by itself.

## Preflight first (no training)

From the repository root:

```bash
python -m aethrion.train_family
```

The default is preflight-only. It validates the exact model-specific dataset
and governance gates. A blocked model is not queued and no optimizer is
initialized. Do not set approval fields merely to make preflight pass.

## Execute only after every relevant approval is complete

```bash
python -m aethrion.train_family --execute --report artifacts/family-report.json
```

The orchestrator runs models sequentially and revalidates each model immediately
before its trainer starts. Each trainer writes its own checkpoint and metrics.
The family supervisor also records:

- `artifacts/training_sessions/<session-id>.json`: atomic latest session
  snapshot, current model, per-model outcome and release count (always zero).
- `artifacts/training_sessions/<session-id>.jsonl`: append-only event journal
  with a flushed and fsynced record for session/model start, finish, block and
  final session outcome.

The journal is local operational state and should be backed up with the
checkpoint artifacts. It contains model IDs, paths and status metadata, not
training examples or API secrets.

## Pause and resume

Press Ctrl+C or send SIGTERM to request a graceful stop. The family supervisor
records an interrupted session. The active trainer receives the same stop
signal and saves after its current optimizer step; the trainer's checkpoint
contains model/optimizer/scaler state where applicable, RNG state, data hashes,
tokenizer/provenance hashes and the training configuration.

Run the same family command again after the environment is available. Each
model preflight detects either `checkpoint_last.pt` or the retained
`checkpoint_last.pt.previous` generation and passes the canonical latest path
to the trainer. The trainer verifies the latest generation and falls back to
the previous generation when the latest file, checksum or safe deserialization
fails. If only the previous generation exists, the family supervisor still
passes the resume argument; it must not silently start from scratch and
overwrite the recovery point. Resume is refused if the approved dataset,
tokenizer, source/approval manifest, model configuration or fixed optimizer
settings have changed. Increasing the configured step horizon is supported;
shortening it is not.

A sudden power loss, kernel kill, machine crash or forced termination cannot
run a signal handler. In that case, the last atomically written checkpoint
(and its verified `.previous` generation) is the recovery point. The family
journal may remain marked `running`; inspect it together with each model's
`training_report.json` and checkpoint checksum before resuming.

## Artifact integrity and portability

Checkpoint payloads are loaded with PyTorch `weights_only=True`; SHA-256
sidecars are checked before deserialization. Both trainer families validate the
checkpoint step, model/optimizer structures, provenance hashes and the full
Python, NumPy, CPU Torch and optional CUDA RNG snapshot before accepting a
generation. A checksum-valid but structurally invalid latest generation is
rejected so the loader can recover from the previous verified generation.
When saving, a previous generation is promoted only if the current checkpoint
passes semantic validation as well as its checksum.

Checkpoint and JSON snapshots are written through temporary files, fsynced,
then atomically replaced. Directory entries are fsynced where the operating
system/filesystem supports it. The specialist and base trainers retain their
own previous verified checkpoint generation.

Keep the entire `artifacts/` directory on persistent storage. A temporary
Kaggle/Colab runtime disk is not a backup: copy artifacts to durable storage
before closing or deleting the runtime. Do not commit large checkpoints,
private datasets or credentials to Git.

## Status interpretation

- `blocked`: governance, dataset, provenance or configuration gate did not pass.
- `training_interrupted_not_released`: paused at a checkpoint; eligible to
  resume only after the same integrity and approval checks pass again.
- `training_completed_not_released`: optimizer finished; model is not released.
- `released_count: 0`: no model release is implied by this training workflow.

Independent held-out evaluation, safety/security review and explicit human
release approval remain mandatory. Development fixtures are for tests only and
must never be promoted into production training corpora.

## Family report verification

After a child trainer exits with code zero, the family supervisor does not infer
success from the exit code or a status string alone. It requires a report
created during the current process attempt, matching the selected model ID and
the explicit `NOT_RELEASED` release state. A completion report must also point
to an existing final checkpoint whose bytes match the report's SHA-256 digest.

The supervisor validates the model-specific report contract:

- CHAT: final step, finite train/validation metrics, finite held-out test loss,
  dataset hash and tokenizer hash.
- Specialist models: positive completed step count, finite validation and
  held-out test losses, plus dataset, tokenizer and manifest hashes.

Missing, stale, malformed, wrong-model, incomplete-metric, missing-checkpoint
and checksum-mismatched reports are recorded as non-success outcomes. A valid
training completion is still `training_completed_not_released`, never a
release. A non-zero trainer exit is a failure even if an old report exists.

For a graceful CHAT interruption, the base trainer writes a separate
`training_interrupted_not_released` report only after its checkpoint save. The
family supervisor verifies that report's freshness, model identity, release
state and checkpoint digest before recording the pause. Specialist interruption
reports follow the same checkpoint-digest requirement. If the report or
checkpoint cannot be verified, the family outcome remains an error state; a
stop signal alone is not proof that a safe checkpoint was written.
