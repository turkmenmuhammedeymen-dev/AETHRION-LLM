# Turkish dataset and training protocol

## Dataset principles

AETHRION does not ship a fabricated corpus. The repository provides a preparation
pipeline and manifest template; the project owner supplies approved source material.
Every source needs a documented origin, license, permitted use, and explicit training
approval. Exclude personal, confidential, private, or unlawfully obtained material.

## External datasets are disabled

Earlier revisions documented a bounded Cosmos Turkish Corpus download. That
workflow is obsolete and must not be used. AETHRION's current data policy is
first-party owner-authored material only; external dataset downloads, gated
dataset access, and third-party corpus imports are disabled.

The legacy command remains only as a compatibility stub:

~~~bash
python -m aethrion.import_hf_dataset --max-records 1
~~~

It exits with an explicit policy error and does not create an output directory,
download data, or contact Hugging Face. Do not enable it or replace the stub
with an external importer without a separately authorized policy change.

Prepare only owner-authored source files kept outside GitHub. Use
[BASE_TRAINING_READINESS.md](BASE_TRAINING_READINESS.md) for the required
source manifest, rights and authorship evidence, privacy/quality review,
human approval, HMAC signing, corpus preparation, and training commands.
Keep training_allowed=false until the exact first-party corpus has been
reviewed and approved. A source manifest, synthetic fixture, or CI test is
not itself evidence of real authorship or rights.

## Preparation

For owner-authored plain text or JSONL, use a manifest that documents each
source and records its actual authorship and rights review. Do not add third-party
URL-attributed records. BASE splitting is provenance-disjoint: each
plain-text source file forms its own group, URL-attributed records are grouped by
exact hostname, and JSONL may provide an explicit `group_id`. At least three
independent groups are required so train, validation and test each receive whole
groups. Keep unrelated sources in separate files or give records explicit group IDs.

The pipeline normalizes Unicode to NFC, collapses whitespace, removes duplicates,
excludes basic email-pattern matches, and enforces length limits. It writes
`corpus-train.txt`, `corpus-validation.txt` and `corpus-test.txt`, hashes each split,
and verifies that no provenance group crosses partitions. The tokenizer is learned
from the train-only file and emits a provenance sidecar binding it to source, corpus,
review and split hashes. The email-pattern filter is only a basic precaution, not a
PII detector. Review audit reports and sample records manually; check Turkish
coverage, source balance, duplication, spelling, encoding, privacy, and license compatibility.

## Train and evaluate

~~~bash
python -m aethrion.tokenizer --train data/processed/corpus-train.txt \
  --corpus data/processed/corpus.txt \
  --output artifacts/tokenizer.json --vocab-size 512 \
  --dataset-manifest data/processed/corpus_manifest.json \
  --source-manifest data/raw/cosmos/sources.json \
  --source-root data/raw/cosmos \
  --corpus-report data/processed/corpus-report.json \
  --source-review data/processed/privacy_review/source_review_template.csv
python -m aethrion.train --config configs/turkish_pilot.json \
  --dataset-manifest data/processed/corpus_manifest.json \
  --source-manifest data/raw/cosmos/sources.json \
  --source-root data/raw/cosmos \
  --corpus-report data/processed/corpus-report.json \
  --source-review data/processed/privacy_review/source_review_template.csv
python -m aethrion.evaluate --checkpoint artifacts/turkish_pilot/checkpoint_last.pt \
  --tokenizer artifacts/tokenizer.json --data data/heldout/approved-eval.txt
~~~

Keep evaluation text disjoint from training data. Build a human-reviewed benchmark
for grammar, reading comprehension, summarization, reasoning, instruction following,
factuality, dialect/register variation, and safety. Record corpus hashes, model and
tokenizer versions, hardware, training steps, and evaluation results.

## Scale and limitations

GitHub Actions CPU tests cannot train a high-capability language model. Large-scale
training requires a carefully sized corpus, capable GPU hardware, experiment tracking,
and repeated independent evaluation. Never claim superiority without controlled,
reproducible benchmark results. The current BPE trainer is a research implementation,
not optimized for multi-billion-token corpora.
