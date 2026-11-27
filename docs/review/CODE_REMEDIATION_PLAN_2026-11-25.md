# CODE corpus remediation plan — 2026-11-25

## Scope and decision

This is a static source/generator review of the CODE pilot corpus. It does not approve training. The CODE manifest remains fail-closed (`training_allowed=false`, `owner_approval=false`). No authored records, prepared NPZ artifacts, tokenizer files, or manifest hashes are changed by this report.

## Confirmed record-level issues

- `code-0013` (`Hesap`): `cek(miktar)` checks only whether the requested amount exceeds the balance. A negative amount passes and increases the balance after subtraction. The constructor also accepts a negative initial balance. Add explicit input invariants and tests for zero, negative, exact-balance, and insufficient-balance cases.
- `code-0037` (Caesar cipher): the implementation rotates only ASCII `A-Z` and `a-z`; Turkish letters such as `ç, ğ, ı, ö, ş, ü` are left unchanged. Either implement a clearly specified Turkish alphabet mode (including case and normalization behavior) or explicitly label the example as ASCII-only and add tests documenting that contract.
- `code-0009` (binary search): the algorithm requires a sorted input sequence, but the record has no docstring or precondition note. Add an explicit sorted-input contract and tests for empty, singleton, present, absent, and duplicate values.

## Generator and provenance consistency

- `data/authored/code/make_code.py` describes its samples as “owner-authored”.
- The current governance/provenance record says the content was drafted by Claude at the project owner's request, and the owner has not individually reviewed all records.
- These statements are not equivalent. The generator docstring should describe the actual creation workflow accurately; it must not imply direct human authorship or review unless that is documented.
- `make_code_more.py` and `make_code_gen.py` append records idempotently by ID. Editing those generators alone will not repair already-written rows because existing IDs are skipped. A deterministic migration/rebuild step is required.

## Split behavior

The current text preparation implementation hashes each row's `group_id` to choose its split. Thus records sharing a group ID stay together. CODE currently has 84 group IDs for 163 rows. Preserve this behavior and add a regression test proving that no group hash appears in more than one split.

## Required remediation sequence

1. Correct the three source examples and their tests in the canonical authored-data generator/migration.
2. Rebuild `data/authored/code/records.jsonl` deterministically from the corrected source; verify stable IDs, unique IDs, group IDs, syntax, and exact source-to-prepared-source equality.
3. Re-run the official CODE preparation pipeline from a clean output directory. Do not hand-edit `source.jsonl`, NPZ arrays, tokenizer, or manifest independently.
4. Recompute and verify source, dataset, and tokenizer SHA-256 values; compare manifest counts to the regenerated files.
5. Add executable behavior tests for every Python sample where practical, language-specific syntax checks for other languages, and explicit unsupported-runtime markers where execution is unavailable.
6. Keep training disabled until the owner completes the required human rights/provenance/privacy/content review and the independent task-level evaluation gate passes.

## Limitations

This review is static and targeted; it is not a full semantic review of all 163 CODE records or a comprehensive security audit of every snippet. A passing syntax or unit-test suite cannot establish dataset rights, factual quality, broad code-generation ability, or production readiness.

## 2026-11-25 implementation status

A source-level remediation has now been prepared on branch `fix/code-corpus-records-and-tests`:

- `code-0009`: the ascending-order precondition is explicit in the record and canonical seed generator.
- `code-0013`: negative opening balances and non-positive withdrawals are rejected in the record and canonical seed generator.
- `code-0037`: the Caesar example now rotates the 29-letter Turkish alphabet, preserving case and leaving punctuation/digits unchanged; the Round 2 generator is updated.
- The Round 2 and Round 3 generator headers now describe generated content accurately rather than implying direct owner authorship or individual review.
- `tests/test_code_corpus_records.py` adds behavior tests and asserts the 163-row / 84-group inventory and the three original group IDs remain unchanged.

**Important artifact state:** the prepared snapshot at `data/specialists/manifests/code/source.jsonl`, its NPZ tensor, tokenizer, and manifest have not been rebuilt in this change. They still represent the previous source snapshot. Training remains disabled. Do not use or release the old prepared CODE artifact as if it contained these fixes. The official preparation pipeline must be run from the corrected authored JSONL into a clean output location, then source/dataset/tokenizer hashes and split counts must be reconciled before any training consideration.

The source edits are not yet merged; CI must pass first. No training was started.
