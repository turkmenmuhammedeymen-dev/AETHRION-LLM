# Cross-product AI integration matrix

**Status: source-audited integration contract; not a claim that the product endpoints are already connected to trained AETHRION models.**

This document maps existing AI entry points in the EMSYLLIS product repositories to the first-party AETHRION model family. The machine-readable contract is `configs/integrations/product_ai_routes.json`; it is validated by `aethrion.product_integration`.

## Non-negotiable runtime contract

- Production model inference is local-only and uses AETHRION-owned architectures, tokenizers/processors and checkpoints.
- No Ollama, OpenAI, Gemini, NVIDIA NIM, Hugging Face Inference Provider, hosted search model, or other third-party model endpoint may be used as a hidden fallback.
- Python, PyTorch, Gradio, Parse Server, browser speech APIs, FFmpeg and ordinary file-format libraries are infrastructure, not model providers. Browser speech APIs are not counted as AETHRION ASR/TTS.
- A product route remains unavailable until every required local model is trained, evaluated, released and hash-verified. A ready checkpoint alone does not prove that the product endpoint adapter, authorization, quotas, data isolation and failure handling have been integrated.
- Tool access is separate from model output. Model-generated text never grants shell, browser, account, domain, repository, email or device permissions.

## Source audit

### Vizyon2026 / AETHRION

Audited repository: `EMSYLLIS/vizyon2026`, default branch `main`.

Existing entry points found in the current source:
- `vizyonAlChat`: standard assistant conversation.
- `vizyonAlVoiceChat`: voice-originated conversation.
- `vizyonAlAgent`: agent-mode wrapper.
- `vizyonAlScreenRead` and `vizyonAlLiveScreenStream`: user-selected screen/camera frame analysis.
- `vizyonAlFileAnalyze`: file analysis.
- `aethrionWorkforceRun`: bounded specialist workforce orchestration.
- `kocChat`: Coach conversation.
- `smartTranslate`: translation.
- `generate3DWithHF`: legacy 3D generation entry point.
- `createVideoClip`: video tooling entry point.

Native mapping:
- Chat, Coach, translation and worker reasoning -> AETHRION-BASE (`chat`).
- Voice input/output -> AETHRION-ASR (`asr`) + BASE (`chat`) + AETHRION-TTS (`tts`) and separately released VOCODER.
- Screen/camera understanding -> VISION + DOC + BASE. The current Vision model is only an image/text embedding similarity architecture; it is not visual question answering, OCR or temporal video understanding. Those tasks remain blocked until their architectures and evaluations support them.
- Files -> DOC + BASE, with VISION/ASR/CODE selected only for compatible MIME types and trained adapters.
- Workforce -> bounded orchestration over the same native specialists, not a claim that every registered role is a separate model.
- 3D -> native 3D model. Current 3D output is a vertex-only point cloud, not a watertight textured mesh; image-to-3D conditioning is not implemented.
- Video -> BASE may plan a script or edit decision list; deterministic media tooling may render supported edits. There is no first-party video-generation model in the current manifest, so generative video must remain unavailable.

### SENTRIX

Audited repository: `EMSYLLIS/SENTRIX`, default branch `main`.

- `sentrixGenerateAiSummary` currently sends aggregate analytics to a configurable OpenAI-compatible provider. Replace the model call with the native BASE model, optionally using RESEARCH for source-grounded evidence summaries and SAFE for output-policy review.
- The summary input is aggregate page views, unique visitors and top pages. It must not infer personal traits or invent security findings.
- `sentrixCreateAssessmentRequest` currently stores an owner-authorized request with `status: requested` and `scannerEnabled: false`. It is not a vulnerability scanner and has no model result to migrate.
- Existing security events are ingested from a separately authenticated source; the AI summary must distinguish observed events from heuristic signals and must never fabricate scan results or scores.
- A future security assessment engine requires an independently scoped, authorized, audited defensive assessment pipeline. A language model alone is not a scanner. SAFE is policy/safety review, not a cyber-vulnerability detector.

### Tdev2027 / TdevKuantum

Audited repository: `EMSYLLIS/Tdev2027`, default branch `main`.

- `tdevAIChat` powers the AI teacher and AI question explanations. Its current backend is configured for NVIDIA and optional OpenAI-compatible providers.
- Native mapping: BASE for tutoring and explanations; DOC for supported documents; VISION for photos only after visual-QA/OCR tasks are implemented and evaluated; RESEARCH for source-grounded research; CODE for code-related educational tasks.
- Official question feeds and verified answer keys remain source-controlled data. The model may explain an answer but must not present generated content as an official key or official curriculum statement.
- A separate education model is not required for the first migration: the existing BASE model can receive education-specific instruction tuning and evaluation. A separately released AETHRION-EDU should be considered only if benchmarks show BASE is insufficient and a distinct model is justified.

## Model status and blocked capabilities

The current native registry contains 11 model programs: chat, code, vision, image, asr, tts, vocoder, 3d, research, document and safety. Their current registry entries are `not_trained`; no production checkpoints exist. Consequently every model-backed route in the machine-readable map is blocked.

The route validator deliberately returns `models_ready_adapter_unverified`, not `ready`, even if model artifacts later pass registry checks. Product endpoint integration still requires its own authenticated end-to-end tests.

## Security and privacy requirements for integration

1. Keep each product's own account/session authorization. Do not share Back4App master keys or reuse a user's session across products.
2. Use a server-side AETHRION gateway with per-product, per-user and per-capability authorization, quotas, timeouts, cancellation and audit events.
3. Never send SENTRIX private analytics, student records, conversation history or uploaded documents to training by default. Training use needs separate purpose, minimization, retention and rights review.
4. Treat webpages, files, images, screen frames, OCR text and worker output as untrusted data. They cannot override system policy or grant tools.
5. Do not run uploaded source code or model-generated shell commands.
6. Return explicit `MODEL_NOT_RELEASED`, `CAPABILITY_UNAVAILABLE`, `RATE_LIMITED` and `INTEGRATION_UNVERIFIED` states. Do not fall back to an external provider.
7. Preserve legacy Cloud Function names while changing their internal model adapter. Existing endpoint names and UI presence do not prove the new adapter is deployed.

## Deployment boundary

This repository owns the model family and route contract. The product repositories keep their own UI, authentication and deployment configuration. No source files in Vizyon2026, SENTRIX or Tdev2027 are modified by this integration-contract change. Each product backend migration must be a separate, tested change with its own deployment and rollback plan.
