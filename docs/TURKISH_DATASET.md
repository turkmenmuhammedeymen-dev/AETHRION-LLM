# Turkish dataset and training protocol

## Dataset principles

AETHRION does not ship a fabricated corpus. The repository provides a preparation
pipeline and manifest template; the project owner supplies approved source material.
Every source needs a documented origin, license, permitted use, and explicit training
approval. Exclude personal, confidential, private, or unlawfully obtained material.

## Bounded public-corpus pilot

The Hugging Face dataset card for `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0`
currently declares CC BY 4.0, 9,075,453 rows, and about 22.8 GB of download data.
It is a candidate source, not a blanket guarantee about every upstream webpage.
Review the card, attribution requirements, privacy, and component rights before use.

For a small streaming sample:

```bash
pip install -e ".[data]"
python -m aethrion.import_hf_dataset --max-records 5000
python -m aethrion.prepare_data --input data/raw/cosmos --manifest data/raw/cosmos/sources.json \\
  --output data/processed/corpus.txt --report data/processed/corpus-report.json
python -m aethrion.tokenizer --train data/processed/corpus.txt \\
  --output artifacts/tokenizer.json --vocab-size 512
```

The importer streams at most 5,000 rows by default and writes attribution metadata.
It does not fetch the entire upstream corpus. The existing BPE trainer is a simple
research implementation and is not yet optimized for a multi-billion-token run.
Use this pilot to validate the pipeline, not to claim high-quality model capability.

## Preparation

```bash
python -m aethrion.prepare_data --input data/raw --manifest data/sources.json \
  --output data/processed/corpus.txt --report data/processed/corpus-report.json
python -m aethrion.tokenizer --train data/processed/corpus.txt \
  --output artifacts/tokenizer.json --vocab-size 8192
```

Review the audit report and sample records manually. Check Turkish coverage, source
balance, duplication, spelling, encoding, privacy, and license compatibility.
The email-pattern filter is only a basic precaution, not a PII detector.

## Train and evaluate

```bash
python -m aethrion.train --config configs/small.json
python -m aethrion.evaluate --checkpoint artifacts/checkpoint_last.pt \
  --tokenizer artifacts/tokenizer.json --data data/heldout/approved-eval.txt
```

Keep evaluation text disjoint from training data. Build a human-reviewed benchmark
for grammar, reading comprehension, summarization, reasoning, instruction following,
factuality, dialect/register variation, and safety. Record corpus hashes, model and
tokenizer versions, hardware, training steps, and evaluation results.

## Scale and limitations

GitHub Actions CPU tests cannot train a high-capability language model. Large-scale
training requires a carefully sized corpus, capable GPU hardware, experiment
tracking, and repeated independent evaluation. Never claim superiority without
controlled, reproducible benchmark results.
