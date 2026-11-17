# AETHRION native multimodal architecture

AETHRION uses separately initialized and separately trained first-party models.
Production inference must not download, call, proxy, distill from or silently
fall back to pretrained third-party models or hosted inference/search APIs.

## Independent model family

- **AETHRION-BASE:** multilingual causal language model; separate pretraining
  and response-only instruction tuning.
- **AETHRION-CODE:** independent causal model for code completion, generation
  and repair.
- **AETHRION-VISION:** image/text dual encoder for visual representation and
  contrastive image-text learning. It is not yet a visual conversational model.
- **AETHRION-IMAGE:** text-conditioned pixel diffusion research model. Its
  current low-resolution U-Net is a research scaffold, not a frontier image
  generator.
- **AETHRION-ASR:** acoustic encoder trained with a CTC transcription objective.
- **AETHRION-TTS:** transcript-to-mel predictor with duration modelling.
- **AETHRION-VOCODER:** separate mel-to-waveform network. TTS is not complete
  or releasable without a separately trained and evaluated Vocoder.
- **AETHRION-3D:** text-conditioned point-cloud diffusion. The present
  implementation exports vertex-only OBJ point clouds; it does not yet create
  complete watertight meshes, topology or textures.
- **AETHRION-RESEARCH:** query/passage dual encoder for local retrieval. A
  production local corpus index, provenance and evidence citation pipeline
  remain to be completed.
- **AETHRION-DOC:** document text/layout encoder with token tagging and
  extractive-QA heads. OCR/layout extraction and complete page understanding
  are not yet complete.
- **AETHRION-SAFE:** separate ALLOW/REVIEW/BLOCK text classifier with reason
  scores. It is not an active safety barrier until trained, calibrated and
  independently evaluated.

## Training and release contract

The repository contains a fail-closed multimodal NPZ dataset validator,
per-model training configurations, a resumable specialist trainer, BASE
instruction tuning, audio WAV/log-mel preprocessing, and shape/loss smoke tests.
Each model has its own checkpoint path and model manifest entry.

A dataset must have recorded source provenance, license and training rights,
source-file SHA-256 values, privacy review, speaker consent where relevant,
deduplication and independent train/validation/test partitions. The model
training flag and owner approval must be exact JSON booleans and explicitly
approved by a responsible human. A training run produces a checkpoint and a
report marked NOT_RELEASED; it never automatically publishes a model.

The native registry resolves a model only when its checkpoint, tokenizer or
processor, training report, evaluation report, model card and dataset hashes
match; explicit release, quality and safety gates must all be true. TTS also
requires the separate AETHRION-VOCODER release.

## Runtime and safety

The production router is local-only. Missing or untrained models return an
explicit unavailable state. No capability tab or Python class may be labelled
ready simply because it exists.

AETHRION-SAFE must be independently trained and evaluated before a serving
mode can claim model-based safety screening. Application-level controls remain
required: authentication, input validation, rate limits, resource limits,
audit logs, abuse tests and human review for consequential operations.

## Current implementation versus completion

The repository now has independent research architectures and training
entrypoints for the model family, but the model weights are not trained and
the data manifests remain fail-closed. Passing unit/forward tests confirms
software mechanics only; it does not demonstrate language, image, speech, 3D,
retrieval, document or safety quality.

Major remaining work includes large-scale lawful datasets, robust high-capacity
architectures, distributed/mixed-precision training validation, model-specific
held-out benchmarks, calibration, OCR/layout tooling, mesh topology and
texturing, research indexing/citations, complete local modality inference
adapters and a production end-to-end release review.
