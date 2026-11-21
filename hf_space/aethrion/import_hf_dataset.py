"""Legacy external-dataset importer intentionally disabled by AETHRION policy.

AETHRION training data must be first-party, owner-authored and human-reviewed.
This command is retained only to provide a clear error for old runbooks; it
must never download, cache, or write third-party dataset records.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External Hugging Face dataset imports are disabled."
    )
    parser.add_argument("--output-dir", default="data/raw/cosmos")
    parser.add_argument("--max-records", type=int, default=5_000)
    parser.add_argument("--split", default="train")
    parser.parse_args()
    raise SystemExit(
        "External Hugging Face dataset imports are disabled by the AETHRION-LLM "
        "first-party data policy. Do not download third-party datasets. Use only "
        "owner-authored material with documented rights, privacy review and "
        "human approval; the training gate remains closed until that review."
    )


if __name__ == "__main__":
    main()
