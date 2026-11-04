# AETHRION-LLM Kaggle pipeline

## One-cell entry point

In a fresh Kaggle notebook, enable a GPU accelerator if available, then run:

```python
from pathlib import Path
import subprocess, sys

REPO = "https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM.git"
BRANCH = "feature/turkish-corpus-pipeline-sync"
ROOT = Path("/kaggle/working/AETHRION-LLM")

if not (ROOT / ".git").exists():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO, str(ROOT)],
        check=True,
    )
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "switch", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT, check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[data]"],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "aethrion.pipeline", "--import-if-missing"],
    cwd=ROOT,
    check=True,
)
```

The importer streams no more than 5,000 records by default. It does not download
the full upstream corpus. The first run is expected to stop safely at the approval
gate because the committed dataset manifest is intentionally set to
`training_allowed: false`.

## What the pipeline does

1. Reports Python, PyTorch, CUDA/GPU, and free disk space.
2. Runs the complete unit-test suite.
3. Optionally streams a bounded source sample when no source manifest exists.
4. Creates a per-domain review CSV with every row initially `PENDING`.
5. Checks the dataset-level manifest, each source manifest entry, and domain review.
6. Only after all gates pass: prepares the corpus, trains the tokenizer, and runs
   the configured pilot training.
7. Runs independent evaluation only if a separately reviewed
   `data/heldout/approved-eval.txt` file exists.

## Required human review before training

Never change the approval flags merely to make the pipeline continue.

- Verify the upstream dataset card and the rights for every original component.
- Review privacy, sensitive information, source terms, attribution, and lawful basis.
- For every exact hostname, record a verifiable HTTPS license/evidence URL, rights
  holder, explicit AI-training permission, attribution requirement, and
  `review_status=APPROVED` in
  `data/processed/privacy_review/source_review_template.csv`.
- Review every source in `data/raw/cosmos/sources.json`; set
  `allow_training=true` only when that particular source is actually approved.
- Have the dataset-level review completed before changing
  `data/processed/corpus_manifest.json` to Boolean `true`.
- Keep a separately collected, reviewed held-out evaluation file. It must not be
  copied from the training corpus.

The pipeline's checks validate recorded decisions; they do not determine copyright,
license compatibility, privacy compliance, or legal permission.

## Pilot configuration and outputs

The default `configs/turkish_pilot.json` is a small engineering pilot (2 layers,
128 embedding width, 500 steps), not a high-capability model recipe. Outputs are
written under `data/processed/` and `artifacts/turkish_pilot/`. Keep generated
corpora, checkpoints, tokenizer artifacts, and private data out of Git.

To run only preflight/tests without importing data:

```bash
python -m aethrion.pipeline
```

To choose another reviewed configuration:

```bash
python -m aethrion.pipeline --config configs/small.json
```

The larger config may require substantially more GPU memory and time. Do not select
it without checking the available hardware and corpus size.
