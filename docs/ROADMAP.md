# AETHRION-LLM roadmap

This roadmap tracks implementation, not claims of model quality. A checkbox
for architecture or code does not mean a trained model is available. No model
may be released until its dataset, checkpoint, held-out evaluation, safety
review and model card pass the registry gates.

## Foundation — AETHRION-BASE

- [x] From-scratch decoder-only Transformer architecture.
- [x] Byte-level BPE tokenizer trained from supplied corpus.
- [x] Corpus preparation, provenance manifest and fail-closed training gate.
- [x] Resumable pretraining, atomic checkpoint and SHA-256 integrity checks.
- [x] Evaluation metric reporting and synthetic training smoke test.
- [x] Response-only instruction-tuning entrypoint and ignored-prompt loss.
- [ ] Human-reviewed lawful multilingual corpus with adequate scale.
- [ ] Full pretraining run on approved data and documented hardware budget.
- [ ] Instruction dataset and tuning run with independent held-out evaluation.
- [ ] Turkish/multilingual factuality, reasoning, hallucination, robustness and safety benchmarks.
- [ ] Production model card and explicit release approval.

The currently tracked general corpus manifest stays training_allowed=false.
The test-only fixtures are not production training data.

## Independent specialist family

### AETHRION-CODE
- [x] Separate randomly initialized causal LM architecture and config.
- [x] Specialist training runner and dataset schema.
- [x] Forward/loss test.
- [ ] Permissively licensed code corpus with provenance and repository deduplication.
- [ ] Code training, unit-test generation, repair benchmark and security review.
- [ ] Instruction/fix evaluation on held-out repositories.

### AETHRION-VISION
- [x] Image/text dual encoder and contrastive objective.
- [x] Dataset schema and shape/loss smoke test.
- [ ] Licensed image-caption and visual-QA dataset.
- [ ] Larger visual encoder, data pipeline, training, retrieval and calibration benchmarks.
- [ ] Visual chat/grounding integration if separately designed and evaluated.

### AETHRION-IMAGE
- [x] Text-conditioned pixel diffusion research architecture and noise-prediction loss.
- [x] Dataset schema and smoke test.
- [ ] Licensed high-quality image-caption and alignment data.
- [ ] Scalable latent diffusion/flow architecture, text conditioning and sampling implementation.
- [ ] Resolution, prompt adherence, composition, safety and memorization benchmarks.

### AETHRION-ASR
- [x] Acoustic encoder and CTC objective.
- [x] Native WAV decoding/resampling and log-mel feature extraction.
- [ ] Consented, licensed multilingual speech corpus and transcript normalization.
- [ ] Training, greedy/beam decoding, WER/CER, dialect/noise/accent evaluation.

### AETHRION-TTS and AETHRION-VOCODER
- [x] Transcript-to-mel predictor and duration objective.
- [x] Separate vocoder architecture and training profile.
- [x] Audio consent gate and WAV output utility.
- [ ] Speaker-consented recordings, validated alignments and multilingual text normalization.
- [ ] Train acoustic model and vocoder independently; verify both hashes and compatibility.
- [ ] Naturalness, intelligibility, pronunciation, speaker-consistency and latency benchmarks.

### AETHRION-3D
- [x] Text-conditioned point-cloud diffusion and vertex-only OBJ export.
- [x] Separate dataset schema and training objective.
- [ ] Licensed text/image/3D pairs with topology/texture metadata.
- [ ] Mesh topology generation, texture/material model, geometry repair and export pipeline.
- [ ] Geometric fidelity, topology, prompt adherence and CAD/game-engine validation.

### AETHRION-RESEARCH
- [x] Local query/passage dual encoder and contrastive objective.
- [x] Approval-gated local source ingestion, chunking, provenance metadata and exact prepared-record SHA-256 binding.
- [ ] Cross-source semantic deduplication, trained retrieval/reranking, citation validation, freshness policy and end-to-end evidence benchmarks.
- [ ] Offline-only search UI and end-to-end citation checks.

### AETHRION-DOC
- [x] Document text/layout encoder, tagging and extractive-QA heads.
- [ ] Local OCR and layout extraction, including tables, reading order and page geometry.
- [ ] Licensed document understanding dataset and separate training/held-out tests.
- [ ] DOCX/PDF semantic and formatting-preserving edit pipeline.
- [ ] Document question-answering, table, layout and OCR benchmarks.

### AETHRION-SAFE
- [x] Separate ALLOW/REVIEW/BLOCK classifier and reason scores.
- [x] Separate manifest/data gate and evaluation requirements documented.
- [ ] Qualified human-reviewed, lawful, privacy-preserving labelled safety dataset.
- [ ] Train, calibrate and independently evaluate category-level false positives/negatives.
- [ ] Adversarial, multilingual and abstention benchmarks; independent safety review.
- [ ] Enable only after documented acceptance criteria and explicit release approval.

## Integration and production
- [x] Local-only model registry with strict artifact/release checks.
- [x] Native-only API routing; no third-party model provider fallback.
- [x] Space deployment path retains existing ZeroGPU hardware.
- [ ] Register each specialist only after checkpoint, processor, reports and release gates are verified.
- [ ] Complete local inference adapters and end-to-end tests for each released model.
- [ ] Concurrency, cancellation, batching, memory, latency and failure recovery tests.
- [ ] Auth, rate limits, quotas, upload isolation, logging and monitoring.
- [ ] Independent security and privacy review.
- [ ] Final release only after every advertised capability meets its own benchmark thresholds.

## Compute reality

Current model classes are research-scale scaffolds. The existing two-T4 Kaggle
runtime and modest pilot configs are not sufficient evidence of parity with
frontier chat, code, image, speech or 3D systems. Before increasing model size,
measure actual GPU memory, storage, step time and projected cost. No paid
hardware or training job should be started implicitly.
