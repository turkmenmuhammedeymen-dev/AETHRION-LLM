# BASE training readiness — evidence-first procedure

This document is specific to the AETHRION-LLM BASE/CHAT model. It does not grant dataset rights or open a training gate.

## Current state

- Dataset gate: `data/processed/corpus_manifest.json` remains `training_allowed: false`.
- No approved BASE corpus is present in the repository.
- The Cosmos Turkish Corpus entry in `docs/DATASET_SOURCE_REVIEW.md` is a candidate only. Its dataset-level CC BY label does not prove rights for each third-party item.
- Do not download or train on that mixed-source corpus unless an item-level rights allowlist, attribution plan and privacy review are completed.
- Do not commit raw corpora, personal data, credentials, or large training artifacts to GitHub.

## What counts as evidence

For every source, retain (outside the public repository if it contains restricted material):

1. Exact dataset name, version/revision and original source URL.
2. The license text or official terms URL for that exact version.
3. Evidence that the rights holder permits machine-learning training for the intended AETHRION use. A general website copyright notice or a dataset card's aggregate label is not enough for mixed-source data.
4. Required attribution and share-alike obligations.
5. Privacy review: exclude personal records, private conversations, secrets, contact details and sensitive identifying data. Record how opt-outs and removal requests are handled.
6. A responsible adult/teacher's review of license, privacy and any consent conditions before data is used.
7. For mixed-source records, per-record provenance and an allowlist; unknown or incompatible records must be excluded, not guessed as approved.

Original text authored specifically for AETHRION can be considered only when the author confirms they created it, have authority to license it for the intended training, and it contains no private or identifying information. Do not copy material from websites, books, school systems, chats or other people and label it owner-authored.

## Required private input layout

Keep source data outside GitHub. On the approved execution environment, the input directory should contain only reviewed source files:

```text
private-data/base/
  sources.json
  owner-authored-text.txt
  # optional: source-review.csv for URL-attributed JSONL
```

Use `data/base_sources.template.json` as a schema example. It intentionally contains no approved entries. Do not change its placeholders to approvals without the evidence above.

A source entry must include:
- unique `source_id`
- relative `path` inside the private input directory
- exact `license` identifier
- `permitted_uses` explicitly including training
- `allow_training: true` only after the human review is complete
- source URL and retrieval date where applicable

For URL-attributed JSONL, every exact hostname also needs an evidence-backed row in `source-review.csv` with HTTPS evidence URL, rights holder, AI-training permission, attribution requirement and `review_status=APPROVED`. Do not use broad wildcard domain approvals.

## Preparation and audit

Only after the actual source review is complete, run from the repository root in an authorized environment with the reviewed private input mounted:

```bash
python -m aethrion.prepare_data \
  --input private-data/base \
  --manifest private-data/base/sources.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --output data/processed/corpus.txt \
  --report data/processed/corpus-report.json
```

For URL-attributed JSONL, add `--source-review private-data/base/source-review.csv`.

The preparation command is expected to fail while the dataset gate is false. It normalizes text, filters email-like records, deduplicates, records source hashes and produces an audit report. It does not prove legal rights; that remains a human review.

After the audit, verify:
- every source hash and the output corpus SHA-256;
- no unreviewed source/domain remains;
- train/validation data are source-disjoint where source groups are available;
- the corpus has enough tokens for both splits and the selected block size;
- the tokenizer is built from this exact approved corpus and its hash is recorded.

## Opening the gate

Only after the responsible owner and adult reviewer have completed the evidence review, update the dataset manifest and source manifest to describe the exact approved corpus. The global gate must stay false if any source is pending, mixed-source rights are unresolved, or privacy checks are incomplete.

Do not treat the template, this checklist, CI fixtures, ZeroGPU smoke tests, or a catalogue entry as approval. Never fabricate license evidence, source hashes, consent or review status.

## BASE training command (not runnable until approval)

The BASE trainer requires all evidence paths and re-validates them before the first training step:

```bash
python -m aethrion.train \
  --config configs/small.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --source-manifest private-data/base/sources.json \
  --source-root private-data/base \
  --corpus-report data/processed/corpus-report.json
```

Add `--source-review private-data/base/source-review.csv` for URL-attributed records. Use `--resume <verified-checkpoint>` only when resuming the same dataset and tokenizer hashes.

## Runtime constraint

The Hugging Face Space ZeroGPU allocation is request-scoped and short-lived. The Space's 11-model preflight is a synthetic forward/backward smoke test, not a persistent training worker and not a BASE training job. GitHub Actions is CI, not a GPU trainer. Do not start an incomplete training run in the preflight endpoint or claim it produced a model checkpoint.

Before launching a real BASE job, the selected runtime must be explicitly configured to run the approved training command and preserve its output/checkpoints securely. If the only available runtime is the short ZeroGPU preflight, full BASE training cannot be launched there.


## Source-disjoint partitions and tokenizer provenance

The current BASE preparation pipeline requires at least three independent provenance groups.
Each plain-text source file is one group; URL-attributed JSONL records are grouped by exact
hostname; non-URL JSONL records may provide a stable `group_id`. Keep records from the same
author, conversation, speaker, document or source together. Do not use per-record unique IDs
as group IDs when records share a provenance relationship.

After the human review opens the dataset gate, `aethrion.prepare_data` writes three
separate files: `corpus-train.txt`, `corpus-validation.txt`, and `corpus-test.txt`.
The validator rebuilds partitions from the current reviewed raw sources, verifies exact
file contents and SHA-256 values, and rejects a provenance group appearing in more than
one partition.

Train the tokenizer only on the approved training partition:

```bash
python -m aethrion.tokenizer \
  --train data/processed/corpus-train.txt \
  --corpus data/processed/corpus.txt \
  --output artifacts/tokenizer.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --source-manifest private-data/base/sources.json \
  --source-root private-data/base \
  --corpus-report data/processed/corpus-report.json
```

Add `--source-review private-data/base/source-review.csv` when URL-attributed records
are present. The command writes `artifacts/tokenizer.json.provenance.json`, binding
the tokenizer hash to the exact combined corpus, train/validation/test partition hashes,
dataset manifest, source manifest and optional domain-review file. The BASE trainer
requires this sidecar and re-verifies it before training and again when resuming.
