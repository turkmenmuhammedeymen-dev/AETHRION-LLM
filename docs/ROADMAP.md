# AETHRION-LLM roadmap

## 0.1 — foundation
- [x] Randomly initialized causal Transformer
- [x] Byte-level BPE tokenizer
- [x] Text/JSONL preparation
- [x] Training, validation, checkpoint/resume
- [x] Terminal inference and optional local API
- [x] Basic tests

## 0.2 — data and evaluation
- [ ] Licensed Turkish-first corpus manifest
- [ ] Deduplication and quality filters
- [ ] Stable train/validation/test splits
- [ ] Perplexity and benchmark harness
- [ ] Training throughput and memory profiler

## 0.3 — instruction following
- [ ] Curated instruction dataset with explicit licenses
- [ ] Supervised fine-tuning pipeline
- [ ] Conversation formatting and stop-token strategy
- [ ] Human-reviewed Turkish evaluation

## 0.4 — reliability and deployment
- [ ] Safety policy and red-team evaluation
- [ ] Factuality and citation evaluation
- [ ] API authentication, rate limits, quotas, monitoring
- [ ] Quantization and optimized inference
- [ ] AETHRION integration adapter behind feature flag
