# AETHRION-SAFE — separate safety model plan

## Purpose and separation

AETHRION-SAFE is a separately versioned classifier intended to support
input/output safety review for AETHRION-LLM. It is not part of the generative
AETHRIONLM weights, is not an instruction-tuned chat model, and must not be
silently substituted for the main model.

The safety classifier is a supporting signal. It does not replace application
policy, deterministic validation, access controls, human review, or independent
testing. A model that has not completed training and evaluation must never be
presented as an active safety barrier.

## Proposed task

Use a compact encoder-style text classifier with:

- A separate model configuration, tokenizer/version record, checkpoint, and
  training manifest.
- Multi-class decision output: `ALLOW`, `REVIEW`, or `BLOCK`.
- Optional multi-label reason tags for policy evaluation and audit reporting.
- Calibrated confidence and an explicit uncertain/abstain path.
- A policy layer that combines the classifier signal with deterministic rules.

The classifier should cover the project's documented safety policy at a
high level. Training examples must be reviewed and labelled by qualified
reviewers; do not create a corpus by collecting private conversations or
unreviewed harmful material.

## Data governance

The safety dataset must have its own independent manifest and approval gate.
Every source needs a documented origin, license, permitted training use,
privacy review, checksum, and explicit approval. Keep train, validation, and
held-out test splits separate. Do not copy evaluation items into training.

Use only owner-authored, consented, public-domain, or otherwise appropriately
licensed and reviewed material. Minimize personal data. Keep raw data and model
artifacts out of Git. Automated checks validate recorded approvals; they do
not establish legal permission or label quality.

## Training and evaluation

Train this model independently from AETHRIONLM. Record the exact dataset hash,
tokenizer hash, model configuration, random seed, optimizer/scheduler,
checkpoint history, and environment. Resume must reject changed data or
tokenizer artifacts.

Before any deployment decision, evaluate:

- False-negative and false-positive rates by policy category.
- Calibration and abstention quality.
- Turkish language quality, spelling variation, and dialect/register coverage.
- Robustness to paraphrase and prompt-injection-style inputs.
- Consistency between input screening and generated-output review.
- Human-review agreement and documented error analysis.

Thresholds must be selected on a held-out validation set and verified on a
separate test set. Do not optimize a single aggregate accuracy score while
ignoring category-level failures.

## Runtime integration

The intended flow is:

1. Validate request shape, size, authentication, and rate limits.
2. Run deterministic policy checks.
3. If an approved AETHRION-SAFE checkpoint is available, classify the input.
4. Apply the documented policy decision; uncertain cases go to review or a
   conservative response path.
5. Generate with the selected capability.
6. Review the output with deterministic checks and, when enabled, the
   independently evaluated safety model.
7. Record minimal, privacy-preserving audit metadata without storing secrets
   or unnecessary user content.

Fail closed if a required safety component is missing for a deployment mode
that claims to enforce it. The ordinary research Space must clearly disclose
when AETHRION-SAFE is not yet trained or enabled.

## Current status

<badge>PLANNED — NOT TRAINED</badge>

No AETHRION-SAFE weights or approved safety-training corpus are currently
included in this repository. This document is an architecture plan, not a
claim that a trained safety model exists. The existing dataset approval gate
must be extended for the safety dataset before training begins.
