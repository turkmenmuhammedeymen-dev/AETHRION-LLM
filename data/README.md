# AETHRION training data

Keep raw/processed corpora, tokenizer files, and model checkpoints out of Git.
Use only owner-authored text, verified public-domain material, or datasets whose
license explicitly permits the intended ML training use.

1. Put UTF-8 .txt or .jsonl files (JSONL needs a string `text` field) under data/raw/.
2. Copy sources.example.json to sources.json and add one manifest entry per file.
3. Record origin, license, permitted use, and explicit training approval.
4. Run the preparation command from the repository root.

The pipeline rejects unlisted/unapproved files, normalizes Unicode and whitespace,
deduplicates records, excludes email-like records, deterministically shuffles, and
writes an audit report with source checksums. These checks do not prove license
validity, Turkish language quality, or absence of personal/copyrighted material.
Human review remains mandatory.
