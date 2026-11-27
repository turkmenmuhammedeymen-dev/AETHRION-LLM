# CPU specialist training: `step=null` / `progress=unknown` root cause (2026-11-26)

## Evidence
- Run 45883024288 (commit 33cc36f): all 10 specialists reported `trainer_ok=True`, `progress={'status': 'unknown', 'step': None}`, `pulled=4`, 60 rounds each.
- A read-only diagnostic run pulled the private artifacts and ran 2 trainer steps without pushing:
  - `code`: `ValueError: Resume model or approved dataset differs from checkpoint.`
  - `vision`: `ValueError: Specialist source/approval manifest changed; refusing unsafe resume.`
- The private artifact repo held only `checkpoint_last.pt[.previous][.sha256]` per model (no `training_report.json`), created before the owner dataset review (#181) changed the approved manifests.

## Root cause
1. The stored checkpoints were trained on older (pre-approval) dataset manifests. The trainer correctly refused to resume.
2. `train_family --execute` exited 0 despite the trainer failure (fixed by #185), so the runner saw `trainer_ok=True`.
3. No training report existed, so progress was `unknown`; the runner repeated 60 no-op sessions (fixed by #185).
4. `subprocess.run(timeout=...)` / SIGTERM stopped only `train_family`, never the child trainer, so a timeout could not produce a clean checkpoint.

## Fix
- `run_session` detects the trainer's stale-resume refusals and reports `stale_checkpoint_incompatible_not_released`.
- With `--fresh-on-stale` (CPU workflow) the stale local files are moved to `artifacts/models/.stale/` (not deleted) and training restarts once from step 0 on the approved data; the new verified checkpoint then replaces the remote one.
- `train_family` forwards SIGTERM/SIGINT to the trainer; the runner uses `graceful_runner`.
- `artifacts/models/` is git-ignored so local checkpoints are never committed.

A completed session is still not a release; evaluation and human approval remain required.
