# Turkish dataset and training protocol

## Dataset principles

AETHRION does not ship a fabricated corpus. The repository provides a preparation
pipeline and manifest template; the project owner supplies approved source material.
Every source needs a documented origin, license, permitted use, and explicit training
approval. Exclude personal, confidential, private, or unlawfully obtained material.

## Bounded public-corpus pilot

The Hugging Face dataset card for ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0
declares CC BY 4.0 for the dataset. This is a candidate source, not a blanket guarantee
about every upstream webpage or component. Review the card, attribution requirements,
privacy, and the rights for each original source before use.

The importer preserves the original URL on each JSONL record. Because the corpus is
mixed-source, a single dataset-level license flag is not enough. Before preparing
URL-attributed JSONL, create and complete a domain review CSV with one row for every
exact hostname. The review must contain a license name, HTTPS evidence URL, rights
holder, explicit AI-training permission (YES), attribution requirement (YES or NO),
and review_status=APPROVED. Unknown, missing, or pending entries block corpus
preparation. Exact hostnames are matched; approval for one host does not
automatically approve its subdomains.

Example:

~~~bash
pip install -e ".[data]"
python -m aethrion.import_hf_dataset --max-records 5000
# Review data/raw/cosmos/sources.json and every original source URL.
# Complete data/processed/privacy_review/source_review_template.csv.
# The dataset-level manifest gate and every exact-host review must be approved.
python -m aethrion.prepare_data --input data/raw/cosmos \
  --manifest data/raw/cosmos/sources.json \
  --source-review data/processed/privacy_review/source_review_template.csv \
  --output data/processed/corpus.txt \
  --report data/processed/corpus-report.json
~~~

The importer streams at most 5,000 rows by default; it does not fetch the full upstream
corpus. Keep allow_training=false until the dataset-level rights review is complete.
The preparation pipeline also requires an evidence-backed approval for every exact
hostname found in URL-attributed JSONL. These checks enforce recorded review decisions;
they do not provide legal advice or independently determine whether a license applies.

## Preparation

For plain text or JSONL without per-record URLs, use a manifest that documents each
source and explicitly approves it. URL-attributed JSONL additionally requires
--source-review as described above.

The pipeline normalizes Unicode to NFC, collapses whitespace, removes duplicates,
excludes basic email-pattern matches, and enforces length limits. The email-pattern
filter is only a basic precaution, not a PII detector. Review audit reports and sample
records manually; check Turkish coverage, source balance, duplication, spelling,
encoding, privacy, and license compatibility.

## Train and evaluate

~~~bash
python -m aethrion.tokenizer --train data/processed/corpus.txt \
  --output artifacts/tokenizer.json --vocab-size 512
python -m aethrion.train --config configs/turkish_pilot.json
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
