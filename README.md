# AETHRION-LLM

**AETHRION-0.2 — from-scratch language-model research project**

This repository builds a decoder-only Transformer from random initialization. It does not download or load pretrained model weights, call Ollama, or depend on third-party inference APIs.

> This is an engineering foundation, not a claim that a capable model has already been trained. The repository starts with a small, reproducible research configuration. Language ability will depend on the quality, licensing, scale, and diversity of the training corpus and on available compute.

## What is included

- Byte-level BPE tokenizer trained from your own corpus.
- Corpus validation and plain-text / JSONL ingestion.
- Decoder-only causal Transformer with tied token embeddings.
- Mixed-precision training where supported, gradient accumulation, clipping, validation, checkpoint/resume.
- Sampling controls: temperature, top-k, top-p, repetition penalty.
- Interactive terminal chat and a small optional HTTP API.
- Unit tests and a tiny synthetic smoke-test path.
- No pretrained weights or hidden model downloads.

## One-command Kaggle workflow

See [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) for the single-cell Kaggle setup and fail-closed pipeline. The first run performs environment checks, tests, and optional bounded data import, then stops at the approval gate while training permission remains false.

## Requirements

Python 3.10+ and PyTorch 2.2+. A CUDA GPU is strongly recommended for meaningful training. Windows 7 is not a supported training environment for current PyTorch releases; use a supported Linux/Windows machine or a temporary cloud GPU runtime. Checkpointing lets training resume after a temporary runtime ends.

Install:

~~~bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
~~~

## Prepare data

Only use text you own, public-domain text, or data whose license explicitly permits machine-learning training. Do not scrape private accounts, personal messages, copyrighted books, or restricted datasets. Keep raw corpora outside Git.

For a bounded pilot import from the public Cosmos Turkish corpus, the dataset card declares CC BY 4.0 at dataset level. That label does not automatically establish rights for every original webpage. The importer preserves each record's URL, and the preparation pipeline requires a separate evidence-backed approval for every exact hostname.

This imports at most 5,000 records as a pilot, not the full upstream corpus. The dataset-level manifest gate and every exact-host review must be approved before URL-attributed data can enter the corpus. Review source attribution, privacy, license compatibility, and sample quality. The current tokenizer is a research implementation; large-scale BPE training needs further optimization before processing multi-billion-token corpora.

For your own approved files, put UTF-8 .txt or JSONL files (with a string text field) in data/raw/, create a source manifest from data/sources.example.json, and run the preparation command with that manifest. The email-pattern filter is not a comprehensive PII detector; human review is required.

The included small config is for smoke tests and experimentation, not ChatGPT-scale quality. Increase model dimensions only after confirming GPU memory and dataset size. Keep data/ and artifacts/*.pt out of Git.

See the Turkish dataset protocol in docs/TURKISH_DATASET.md.

## Chat

After training has produced a checkpoint:

~~~bash
python -m aethrion.chat --checkpoint artifacts/checkpoint_last.pt --tokenizer artifacts/tokenizer.json
~~~

Optional local API:

~~~bash
pip install -e ".[api]"
AETHRION_CHECKPOINT=artifacts/checkpoint_last.pt AETHRION_TOKENIZER=artifacts/tokenizer.json uvicorn aethrion.api:app --host 127.0.0.1 --port 8000
~~~

The API binds to localhost by default in the example. Do not expose it publicly without authentication, rate limits, request-size limits, monitoring, and a safety review.

## Tests

~~~bash
python -m unittest discover -s tests -v
~~~

## Project status

Version 0.2 is the training/inference foundation. It is not instruction-tuned, safety-aligned, or production-hardened. A raw next-token model can produce unreliable, biased, or unsafe text. Do not treat its output as factual or professional advice. Before real users: build evaluation sets, measure hallucination and harmful-output rates, add policy training and robust serving controls, and get independent review.

## Roadmap

1. Verify tokenizer, data pipeline, model forward pass, and checkpoint resume.
2. Train a small baseline on a licensed Turkish corpus.
3. Build held-out language-model evaluation and contamination checks.
4. Add instruction tuning with curated, licensed examples.
5. Evaluate safety, factuality, Turkish quality, and robustness before deployment.
6. Optimize inference and integrate with AETHRION only through a separately reviewed adapter.

## License

No license is granted to third-party training data by this repository. Add an explicit project license before redistributing code or model weights.
