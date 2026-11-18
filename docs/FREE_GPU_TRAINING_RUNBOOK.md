# AETHRION free GPU training runbook

This runbook describes an optional, no-paid-compute path for running already
approved AETHRION training jobs. It does not approve datasets, create model
weights by itself, or bypass the repository's fail-closed data gates.

## What is free and what it can do

- GitHub Actions: CI and CPU tests. It is not a persistent GPU training worker.
- Hugging Face Space ZeroGPU: interactive, request-scoped GPU use. It is not
  intended for long-running model-family training.
- Kaggle Notebooks: optional free GPU sessions, subject to availability,
  queueing and changing quotas. Save checkpoints to /kaggle/working and
  download them before the session ends.
- Google Colab Free: another optional notebook runtime; GPU access and runtime
  limits are dynamic and not guaranteed.

Do not buy hardware, credits, datasets or subscriptions for this workflow.
Do not use a paid accelerator or billing-linked cloud project.

## Before any real training

1. Obtain the exact source release and retain its license, attribution and
   provenance evidence.
2. Have a responsible adult/teacher review the source terms, privacy, creator
   rights and any required speaker consent. A catalogue license label alone
   is not a completed rights or consent review.
3. Keep the source and prepared dataset private while reviewing it. Never put
   private conversations, personal records, secrets or identifying data into
   the training corpus.
4. Prepare the model-specific NPZ and manifest using the commands in
   data/specialists/README.md. The tools deliberately write all approval
   flags as false.
5. Only the responsible owner may update those flags after actual review.
   Never set a flag to true just to make training start.

## Kaggle setup (optional free GPU)

1. Open Kaggle Notebooks at https://www.kaggle.com/code and create a new notebook.
2. Open Notebook Settings and select Accelerator → GPU if a free GPU is
   offered. Hardware type, queueing, session length and weekly quota can vary.
3. Use a read-only, fine-grained GitHub token for this private repository only.
   Store it in Kaggle Secrets as AETHRION_GH_READ_TOKEN; do not paste it into
   a notebook cell, commit it, or share it in chat. If you cannot safely create
   or store a token, download the repository ZIP from GitHub while signed in
   and add it as a private Kaggle input instead.
4. Add only approved, private dataset inputs to the notebook. Do not publish
   raw source audio, images, documents or code datasets.
5. Run the preflight first. It does not start training:

    python -m aethrion.train_family --models code --report /kaggle/working/preflight-code.json

A blocked result is expected until the exact dataset manifest, source hashes,
license/privacy review and owner approvals are complete.

## Training after preflight is genuinely ready

Run one model at a time, not the entire family in one free session:

    python -m aethrion.train_family --execute --models code --report /kaggle/working/train-code.json

Replace code with one of the supported model IDs only after that model's
preflight succeeds. The trainer resumes from a verified checkpoint when one
exists. Keep /kaggle/working outputs and download them before the runtime
ends. Do not commit checkpoints or raw datasets to GitHub.

The sequential family command exists, but a free session is unlikely to be
sufficient for all eleven models. Use individual models and preserve each
checkpoint between sessions. A successful training report still says
NOT_RELEASED; held-out evaluation, calibration where applicable, safety
review, artifact verification and explicit release approval are separate gates.

## Current preparation adapters

- BASE: existing audited corpus preparation and tokenizer pipeline.
- CODE, RESEARCH, DOCUMENT, SAFE: python -m aethrion.specialist_prepare.
- ASR, TTS, VOCODER: python -m aethrion.specialist_prepare_audio.
- VISION, IMAGE, 3D: python -m aethrion.specialist_prepare_visual.

Important limits:
- DOCUMENT needs OCR text, normalized boxes and labels supplied by a reviewed
  OCR process; this adapter is not itself an OCR engine.
- TTS needs token-level durations from a reviewed alignment process.
- SAFE labels must be supplied by human review; the preparation tool does not
  invent or infer safety labels.
- All preparation tools keep training, owner, privacy, source-license and
  speaker-consent approval flags false.

## Stop conditions

Stop before training if a source's license or consent scope is unclear, the
manifest validator fails, a hash differs, the runtime is out of storage, or a
checkpoint cannot be verified. Never bypass the gate, lower the validation
thresholds, or label a model trained without its actual checkpoint and
evaluation evidence.


## Hugging Face ZeroGPU architecture preflight

The Space includes a **Run 11-model ZeroGPU preflight** button. It runs a
short, synthetic forward/backward smoke test for BASE/CHAT and all ten
specialists on the GPU allocated to that request. It reports the device,
peak allocated VRAM, parameter counts, finite loss and gradient checks.

This is an architecture/hardware test only:
- It uses disposable synthetic tensors, not production or user data.
- It does not call optimizer.step(), create checkpoints, or update the model
  registry.
- It does not train, fine-tune, or release any model.
- All eleven models remain NOT_RELEASED after a successful preflight.

The test is useful for catching import, model-construction, CUDA allocation,
forward-pass, loss and backward-pass problems before an approved dataset is
available. A successful result is not evidence of model quality or production
training readiness. The ZeroGPU request is short-lived; it is not a persistent
training worker.
