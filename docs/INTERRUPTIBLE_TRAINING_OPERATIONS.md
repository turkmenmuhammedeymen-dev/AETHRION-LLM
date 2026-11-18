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

The supervisor validates the model-specific report contract and opens
the checkpoint with the same safe, verified loader used by the trainer:

- CHAT: report `step` must exactly equal the checkpoint's zero-based step.
- Specialist models: report `steps` must equal checkpoint step plus one, and
  the checkpoint model ID must match the selected specialist.
- Both families: report dataset/tokenizer hashes must match the checkpoint;
  specialist manifest hash must also match. Metrics must be finite, and held-out
  test loss is mandatory for completed (not interrupted) reports.


A report/checkpoint step or provenance mismatch is a non-success outcome even
when the checkpoint file hash matches. This catches stale reports, mismatched
artifacts and incomplete session handoffs before a family result is recorded.

Each family child process receives the parent session ID and a unique per-model
attempt ID. Both identifiers are written into every checkpoint generation and
trainer report. Reports also carry a creation timestamp and checkpoint-generation
counter. The family supervisor requires exact session/attempt identity and
generation equality between report and checkpoint, and rejects report timestamps
older than the current process attempt. The append-only JSONL journal records
attempt start and, after successful verification, the report timestamp,
checkpoint generation and checkpoint digest. This binds the session snapshot,
event history, report and checkpoint to one execution attempt.

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


## Hash-chained session journal and crash recovery audit

New sessions use a SHA-256 hash chain in every JSONL event. Each record binds the canonical event payload to the previous event digest. The auditor checks the full chain, one session ID, unique attempt IDs and the lifecycle order (attempt prepared -> process started -> verified terminal event). It refuses to append to a damaged or legacy unhashed journal. A truncated final line is preserved and reported as corruption; it is never silently discarded.

The atomic session snapshot records the verified journal event count and final digest. The final run report carries the same values. Before recovery, both files must agree with the journal. Use this read-only command:

```bash
python -m aethrion.train_family --root . --audit-session SESSION_ID
```

A valid audit does not itself start training. `checkpoint_verification_required` means a process-start event has no terminal event: inspect the model's current and previous checkpoint generations and their report/session/attempt metadata before any resume. `safe_to_requeue_after_preflight` means an attempt was prepared but the child process did not start. `safe_to_resume_after_preflight` means the supervisor recorded a graceful interruption. Every case must still pass the ordinary data, provenance, configuration and governance preflight. An invalid chain, legacy unhashed journal, missing snapshot or snapshot/hash mismatch requires manual review and must not be auto-repaired or resumed.

This audit reconstructs verified session state; it does not claim that an interrupted optimizer step was completed, and it never changes production training approvals or release state.


The audit result also includes `reconstructed_session`: the verified snapshot rows are reconciled with each attempt's last journal state. A process-started row is explicitly tagged `checkpoint_verification_required`; a prepared-but-not-started row is tagged `safe_to_requeue_after_preflight`; terminal attempts remain terminal records. The reconstruction is descriptive only (`automatic_resume: false`) and cannot spawn a trainer or mark a model complete.


## Exact checkpoint/report/journal cross-validation

The family outcome verifier now compares the report's checkpoint SHA-256 against both `checkpoint_last.pt` and `checkpoint_last.pt.previous`. It chooses only the exact byte generation named by the report; it does not silently substitute a newer checkpoint. The selected generation must pass its own checksum sidecar and semantic deserialization checks, and its session ID, attempt ID, generation number, step and data/tokenizer/manifest provenance must match the report.

The read-only `--audit-session` path performs the same cross-check for every recorded attempt. For terminal attempts it additionally requires the JSONL `model_finished` event's checkpoint generation, checkpoint digest, report timestamp and terminal outcome to match the report exactly. A damaged latest file can therefore be recovered only when the report names the intact previous generation and all lineage checks agree. Missing reports, mismatched attempts, generation/hash disagreement or provenance mismatch return a manual-review result; no checkpoint is rewritten and no training is resumed.

Regression tests cover a corrupt latest checkpoint with an intact previous generation, a report that points to the wrong generation, journal/report generation disagreement, and exact previous-generation binding. CI execution remains necessary before this code is considered validated.
