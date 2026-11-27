# CODE source repair and prepared-artifact status — 2026-11-25

## Source corrections prepared

Three targeted CODE examples were repaired in `data/authored/code/records.jsonl` and their canonical seed generators:

- `code-0009`: added an explicit ascending-sorted-input precondition for binary search.
- `code-0013`: reject a negative initial balance and reject zero/negative withdrawals before balance mutation.
- `code-0037`: use the Turkish 29-letter alphabet with separate upper/lower alphabets; preserve punctuation and digits; support negative and multi-turn shifts via modulo.

The generator source was updated for the matching seed examples. The Round 2 Caesar generator was updated so rerunning it does not reintroduce the ASCII-only implementation. Round 2 and Round 3 generator descriptions were corrected to say the templates were generated at the owner's request and are not individually human-reviewed.

## Regression tests

Added `tests/test_code_corpus_records.py` to:
- assert 163 unique records and 84 group IDs remain;
- preserve the three affected group IDs;
- execute the three corrected snippets and test relevant boundary behavior.

## Prepared artifact warning

The existing prepared CODE snapshot remains stale relative to the corrected authored JSONL:
- `data/specialists/manifests/code/source.jsonl` is not updated by this source-only patch;
- `dataset.npz`, tokenizer, and manifest hashes have not been regenerated.

This report intentionally does not claim the prepared dataset is repaired. The official `aethrion.specialist_prepare` pipeline must run from the corrected authored JSONL into a clean output directory. Then the new source copy, NPZ, tokenizer, split counts, and SHA-256 values must be reconciled. The training gate remains false and the old artifact must not be used for training or release.

## Limits

These tests cover only three targeted examples and inventory lineage. They do not validate all 163 snippets, other programming languages, security, dataset rights, factual correctness, or production code-generation capability. CI is still required for this branch. No training was started.
