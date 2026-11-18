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
- **AETHRION-RESEARCH:** query/passage dual encoder for local retrieval. An
  approval-gated local index builder now binds prepared records to an exact
  SHA-256 and source provenance. A trained retriever, semantic deduplication,
  reranking, citation validation and freshness policy remain incomplete.
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
texturing, trained research retrieval/reranking and citation validation,
complete local modality inference adapters and a production end-to-end release review.

## Main assistant and specialist delegation

The user-facing assistant entry point is **AETHRION-CHAT**. A separate
`AssistantOrchestrator` runs before the CHAT model: it classifies the latest
user turn using conservative, deterministic task cues and delegates a clear
single-purpose request to the matching native specialist. General, uncertain,
or multi-intent requests remain with CHAT. A specialist that is not registered
as a verified release returns an unavailable error; the orchestrator never
silently substitutes CHAT for a failed specialist.

| Model | Responsibility in the routed system |
|---|---|
| CHAT | Default conversation, explanation and ambiguous requests |
| CODE | Code generation, debugging and repair |
| VISION | Image/text representation and comparison; requires an uploaded image |
| IMAGE | Text-to-image generation |
| ASR | Audio transcription; requires an uploaded audio file |
| TTS | Text-to-speech synthesis; requires a text payload and both TTS + VOCODER releases |
| VOCODER | Internal TTS dependency: converts predicted mel features to waveform; not a user-facing route |
| 3D | Text-conditioned point-cloud generation |
| RESEARCH | Retrieval over the approved local research index |
| DOCUMENT | Uploaded document analysis; requires a bounded document upload |
| SAFE | Mandatory ALLOW/REVIEW/BLOCK policy gate before and after specialist text operations; not a user-selectable task |

The same orchestration is exposed through the protected `POST /assistant` API
and the Hugging Face Space's main chat interface. Explicit specialist endpoints
remain available for file-bearing workflows. The general assistant endpoint
rejects caller-supplied filesystem paths; image/audio/document inputs must use
their dedicated bounded upload interfaces.

The current classifier is deliberately rule-based rather than asking CHAT to
emit routing JSON. This is a temporary reliability choice: the CHAT checkpoint
is not yet trained/released, and an untrained language model cannot safely be
trusted to select tools. Once CHAT is trained, an optional model-assisted
classifier may propose a route, but deterministic validation, capability
availability checks, SAFE gates and explicit user confirmation for ambiguous
or consequential actions must remain authoritative.

**Current limitation:** this is implemented orchestration software, not proof
that any specialist is ready. The production model manifest still marks all
11 models `not_trained`; therefore requests will stop at the availability
gate until the relevant native model(s) are trained, evaluated and released.
