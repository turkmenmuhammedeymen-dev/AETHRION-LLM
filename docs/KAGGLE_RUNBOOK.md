# AETHRION-LLM Kaggle pipeline

## One-cell entry point

In a fresh Kaggle notebook, enable a GPU accelerator if available, then run:

```python
from pathlib import Path
from kaggle_secrets import UserSecretsClient
import os, subprocess, sys

REPO = "https://github.com/EMSYLLIS/AETHRION-LLM.git"
BRANCH = "hardening/training-approval-gate"
ROOT = Path("/kaggle/working/AETHRION-LLM")

# Add a Kaggle Notebook Secret named AETHRION_GITHUB_TOKEN.
# Use a fine-grained GitHub token with read-only access to this private repository.
token = UserSecretsClient().get_secret("AETHRION_GITHUB_TOKEN")
if not token:
    raise RuntimeError("Kaggle Secret AETHRION_GITHUB_TOKEN is missing.")

askpass = Path("/kaggle/working/.aethrion-git-askpass.sh")
askpass.write_text(
    '#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "x-access-token" ;; '
    '* ) printf "%s\\n" "$AETHRION_GITHUB_TOKEN" ;; esac\n',
    encoding="utf-8",
)
askpass.chmod(0o700)
git_env = os.environ.copy()
git_env["AETHRION_GITHUB_TOKEN"] = token
git_env["GIT_ASKPASS"] = str(askpass)
git_env["GIT_TERMINAL_PROMPT"] = "0"

if not (ROOT / ".git").exists():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO, str(ROOT)],
        env=git_env, check=True,
    )
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=ROOT, env=git_env, check=True)
    subprocess.run(["git", "switch", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT, env=git_env, check=True)

# Do not print the token or the authenticated Git environment.
del token, git_env
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[data,api]"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "aethrion.pipeline", "--import-if-missing"], cwd=ROOT, check=True)
```

The importer streams at most 5,000 records by default. It does not download the
full upstream corpus. The first run is expected to stop at the approval gate
because `training_allowed` is intentionally false.

## Pipeline behavior

1. Reports Python, PyTorch, CUDA/GPU, and disk.
2. Runs unit tests.
3. Optionally streams a bounded review sample if source files are missing.
4. Creates a per-domain CSV with every row initially PENDING.
5. Checks dataset, source, and per-domain approvals.
6. Only after every approval passes: prepares data, trains tokenizer and runs
   the configured pilot training.
7. Evaluates only on a separate reviewed held-out file, reporting loss,
   perplexity, tokens/second, parameter count and peak GPU memory.

## Checkpoint recovery and safe interruption

- Checkpoints are written atomically as `checkpoint_last.pt` with a matching `.sha256` integrity file.
- One previously verified checkpoint generation is retained for recovery if the newest file is incomplete or corrupted.
- `progress.json` records the current state; `metrics.jsonl` records periodic training/validation metrics and throughput.
- Ctrl+C or a termination signal requests a safe stop after the current optimizer step and forces a checkpoint save. Resume with the same data, tokenizer, and approval manifests using `--resume`.
- A hard power loss or forced kill cannot be guaranteed to run a signal handler; the last completed checkpoint remains the recovery point.

## Human review before training

Never change approval flags just to make the pipeline continue. Verify rights
for every upstream component, privacy, source terms and attribution. Review each
hostname and source entry. Keep `data/processed/corpus_manifest.json` false
until a documented human review has approved training. Keep a separate
`data/heldout/approved-eval.txt`; it must not be copied from training data.

Automated checks validate recorded decisions; they do not establish copyright,
license compatibility, privacy compliance or lawful permission.

## Provider adapters

See [PROVIDER_SETUP.md](PROVIDER_SETUP.md) for environment variables and the
API startup instructions. Provider credentials are not stored in Git. Provider
adapters are opt-in; live tests may consume paid credits and are not run by CI.

## Pilot limitations

The default `configs/turkish_pilot.json` is a small engineering pilot
(2 layers, 128 embedding width, 500 steps), not a high-capability model recipe.
Generated corpora, checkpoints, tokenizer artifacts and private data stay out
of Git. The API has no built-in authentication or rate limiting: keep it local
or behind authenticated HTTPS.
