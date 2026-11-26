# BASE candidate preparation — 2026-11-26

## Purpose

Add a reproducible, fail-closed preparation path for the existing BASE JSONL files. The tool creates a candidate snapshot for inspection; it does not replace the disabled legacy approval/preparation flow and does not train a model.

## Implementation

- `scripts/prepare_base_candidate.py` reads every `*.jsonl` file in `data/authored/base/` and creates the fail-closed train/validation/test candidate.
- `scripts/prepare_base_chat_candidate.py` learns a candidate byte-BPE tokenizer from the BASE train split only, writes a train-only `corpus.txt` and tokenizer provenance, then prepares CHAT instruction NPZ data using that exact tokenizer.
- Each row must be a JSON object with non-empty UTF-8 `text`. Candidate generation fails closed if it detects email-like strings, Turkish mobile-number patterns, IPv4-like strings, or credential literals.
- Exact normalized duplicates are detected using Unicode NFKC, case-folding, and collapsed whitespace. Stored text is preserved apart from trimming its outer whitespace.
- Because current BASE rows do not carry stable group IDs, non-dialogue rows use the source filename as a provisional group. CHAT-derived dialogue rows use the exact normalized assistant-response hash as their group, preventing the same answer from crossing splits without merging all rows from entire dialogue files. Exact normalized duplicate rows union their groups.
- The deterministic group split targets 80/10/10 by group count, with at least one eligible group in each split. Group sizes can make record-level percentages differ.
- The output is written to `data/processed/base_candidate/`: `train.txt`, `validation.txt`, `test.txt`, `records.jsonl` (exact reviewable records with source line and split), and `manifest.json`. The follow-up stage adds `corpus.txt`, `tokenizer.json`, `tokenizer_candidate_provenance.json`, and a nested `chat_candidate/` containing the CHAT NPZ, matching tokenizer copy, source snapshot, and fail-closed manifest. Source-file hashes, split hashes, record-level source locations, group assignments, and duplicate counts are recorded.
- The writer replaces only its five exact output filenames and never deletes unrelated files.
- `.github/workflows/base-candidate-preflight.yml` runs the tests, builds the candidate, and uploads it as `aethrion-base-candidate-preflight`.

## Safety and release state

The generated manifest explicitly keeps `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false`. It does not modify `data/processed/corpus_manifest.json`, any official specialist manifest, or any model checkpoint. The candidate tokenizer is learned only from the BASE train split and is not installed as the official tokenizer. The CHAT candidate uses that exact candidate tokenizer; both candidate manifests keep all approval flags false. On a push to `main`, the workflow stores BASE and CHAT candidates under commit-specific `candidates/base/` and `candidates/chat/` paths in the private `vizyon2027/aethrion-artifacts` repository. No model-weight training or release is triggered.

## Limitations requiring review

- Filename-based grouping for non-dialogue rows is a provisional fallback. A stronger future format should store stable source/group IDs in each BASE row.
- Deduplication only catches exact normalized duplicates, not paraphrases or factual contradictions.
- Structural validation does not establish factual correctness, copyright/license status, privacy clearance, or quality. The script excludes malformed ordinal-month phrases and generic animal-property templates from train/validation/test, caps identical CHAT assistant responses at three examples, and retains every source row with inclusion/exclusion reasons in the review JSONL. These conservative heuristics do not establish semantic quality or approve the remaining records.
- The generated artifacts must be inspected and their hashes pinned before any human review decision. Tokenizer creation is preprocessing, not model-weight training or approval. This PR does not approve the corpus or the CHAT dataset.
- CHAT/BASE integration remains blocked until the exact candidate snapshot and its source lineage receive the required review.

## Verification status

The PR adds focused tests for duplicate-group union, repeated-answer split leakage, response caps, non-empty splits, SHA-256 consistency, fail-closed flags, PII-pattern blocking, and malformed JSONL rejection. After this PR is merged, the push-triggered workflow will store the BASE and CHAT candidates in the private HF artifact repository under commit-specific paths; PR runs only create GitHub Actions artifacts. A BASE-only artifact is already available from run `45852483911`. The extended tokenizer/CHAT preparation stage completed successfully in Actions run `45853975805`; artifact `15134482032` was downloaded and inspected. The candidate hashes and split counts are recorded below. The run was a PR preflight, so its private-HF upload step was intentionally skipped; the push-to-main workflow will perform that upload after merge.


## First generated artifact — 2026-11-26

GitHub Actions run [45852483911](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/actions/runs/45852483911) completed successfully. Its artifact is `aethrion-base-candidate-preflight` (artifact ID `15133758612`, archive SHA-256 `156aea382e822db057a150d2b321b391e35c03fda7e89793ac37212b78638d08`).

Verified by downloading and inspecting that exact artifact:

- 93 source JSONL files; 1,792 input rows; 1,792 unique full-text rows; zero exact normalized full-record duplicates.
- 1,006 rows included in candidate splits; 786 retained in the review JSONL but excluded from train/validation/test.
- Exclusions: 722 excess repeated-response rows (response cap = 3), 56 malformed ordinal-month rows, and 36 generic animal-property template rows.
- Candidate split: train 807 rows / 369 groups; validation 98 / 46; test 101 / 46.
- 432 unique included dialogue-response groups; zero identical assistant-response groups crossing splits.
- Review JSONL count and SHA-256 match the manifest; all three split-file SHA-256 values match; every included row's split agrees with its group assignment.
- The fail-closed privacy pre-screen completed with zero email-like, Turkish-mobile-like, IPv4-like, or credential-literal matches across all 1,792 rows; a match would abort candidate creation.
- `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false` remain unchanged.

Split SHA-256:
- train: `b01505a8d9561ec1a665bcb438762068c720cc20744baf255b712ada0073c7cc`
- validation: `88391a79372d72451bc98a86f60959628cc645998e65d007934bfde13328156a`
- test: `baeb6b56916306510e7043c53bc2a31d3a2d6ef006301c02ae524f2fc0e2e92d`

This is a reproducible structural candidate, not an approved training corpus. The 1,242 quality-flag count includes records with overrepresented responses even where the first three examples were retained. Factual review, semantic quality, provenance/rights and full privacy clearance remain open. The pattern scan is a limited pre-screen, not a guarantee that all personal or confidential information has been detected.


## BASE tokenizer and CHAT candidate preflight — 2026-11-26

GitHub Actions run [45853975805](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/actions/runs/45853975805) succeeded. Artifact ID: `15134482032`; archive SHA-256: `ad0e9c7afb69287ac05399b620c0ca083bf95f59dc0e451d1bbfd895ee26d6f3`.

Verified by inspecting the exact artifact:

- BASE tokenizer was learned only from the 807-row BASE train split; validation and test partitions were not used.
- BASE candidate split: 807 train / 98 validation / 101 test; 1,006 candidate rows.
- BASE candidate train-only corpus SHA-256: `f0b110194bf029ff6bbdbe1a0b46487fc2be8a7567dbdb6ccc533d88fdbd8a13`.
- Candidate tokenizer SHA-256: `6ad7ba13563cd7bf97290c353c2db12be1ac63d5c5dd8154f29c01ff53a7c1d9`; configured vocabulary ceiling 2,048 (actual vocabulary size recorded in tokenizer file).
- CHAT candidate: 1,989 rows; 1,610 train / 141 validation / 238 test; 441 group IDs; no group ID crosses splits.
- CHAT dataset SHA-256: `0f6ae1cbdea27dc7d0dd450220f13f54c97bdf7bae61f8eb8af8e492f003cd77`.
- CHAT source SHA-256: `5b1edbf0a877a17d7501287be349ce065a5dabe0dc31491ec127582688f5ed57`.
- The CHAT dataset tokenizer SHA-256 exactly matches the BASE candidate tokenizer SHA-256.
- Both BASE and CHAT candidate manifests keep `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false`.

This verifies preprocessing integrity and group-split consistency only. It does not establish factual correctness, rights, privacy clearance, consent, or approval for model-weight training. The private-HF upload did not run during this PR event; it is configured for the subsequent main-branch push.
