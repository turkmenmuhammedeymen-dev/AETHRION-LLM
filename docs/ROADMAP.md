# AETHRION-LLM roadmap

## 0.1 — research foundation

- [x] Randomly initialized decoder-only causal Transformer
- [x] Byte-level BPE tokenizer
- [x] Text/JSONL corpus preparation with deduplication and audit reports
- [x] Single-GPU training loop, validation, and checkpoint/resume
- [x] Sampling, terminal inference, and optional HTTP API
- [x] Unit and synthetic smoke-test foundations

## 0.2 — data governance and training reliability

- [x] Dataset-level and per-source approval manifests
- [x] URL-hostname review for URL-attributed corpora
- [x] Corpus checksums and held-out evaluation separation
- [ ] Enforce all approval and corpus-integrity checks in direct training CLI
- [ ] Verify checkpoint integrity, safe resume, interrupted-session recovery,
      and durable progress reporting across supported runtimes
- [ ] Run the complete Python matrix and training resume tests successfully

Training remains blocked while `data/processed/corpus_manifest.json` has
`training_allowed: false`. Never change this flag just to make a run proceed.

## 0.3 — Turkish baseline and model quality

- [ ] Approve a sufficiently large, licensed Turkish-first corpus
- [ ] Review data quality, privacy, deduplication, attribution, and contamination
- [ ] Improve tokenizer throughput and evaluate vocabulary/context choices
- [ ] Train a small reproducible baseline and preserve resumable checkpoints
- [ ] Add two-GPU training for the available two-T4 Kaggle environment
- [ ] Evaluate on an independent held-out Turkish set and publish reproducible metrics
- [ ] Add human-reviewed grammar, comprehension, summarization, reasoning, and factuality tests

The current pilot configuration is an engineering smoke/pilot setup, not a
high-capability production model. No trained model-quality claim is made.

## 0.4 — instruction tuning and separate safety model

- [ ] Build a curated, licensed instruction dataset and supervised fine-tuning pipeline
- [x] Document the independent AETHRION-SAFE classifier architecture
- [ ] Create a separate safety dataset manifest and approval gate
- [ ] Train and calibrate AETHRION-SAFE independently from AETHRIONLM
- [ ] Evaluate false negatives, false positives, abstention, Turkish coverage,
      robustness, and human-review agreement
- [ ] Add policy-engine integration only after independent safety evaluation

AETHRION-SAFE is planned and has no trained weights yet. It must not be described
as an active safety barrier until its dataset, training, and evaluation are complete.

## 0.5 — multimodal capability adapters

- [x] Explicit capability router and provider registry
- [x] Code, image, 3D, research, file-analysis, and turn-based audio adapters
- [x] Gradio Space interface for the seven capability groups
- [ ] Restore and verify the live Hugging Face Space runtime
- [ ] Run mocked integration tests for every adapter and file operation
- [ ] Run opt-in live provider tests and validate real output artifacts
- [ ] Add provider-specific quotas, cost controls, retries, and observability

Provider adapters are not the same as locally trained AETHRION models. Live
provider functionality remains unverified until real end-to-end tests pass.

## 0.6 — production serving and AETHRION integration

- [ ] Add API authentication, rate limits, quotas, request logging, and monitoring
- [ ] Harden public serving, upload validation, isolation, and secrets handling
- [ ] Optimize inference and evaluate quantization
- [ ] Build the separately reviewed AETHRION Gateway adapter
- [ ] Integrate behind a feature flag and verify rollback behavior
- [ ] Publish model card, dataset provenance, evaluation report, and release checklist
