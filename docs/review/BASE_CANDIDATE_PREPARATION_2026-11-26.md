# BASE candidate preparation — 2026-11-26

## Purpose

Add a reproducible, fail-closed preparation path for the existing BASE JSONL files. The tool creates a candidate snapshot for inspection; it does not replace the disabled legacy approval/preparation flow and does not train a model.

## Implementation

- `scripts/prepare_base_candidate.py` reads every `*.jsonl` file in `data/authored/base/`.
- Each row must be a JSON object with non-empty UTF-8 `text`. Candidate generation fails closed if it detects email-like strings, Turkish mobile-number patterns, IPv4-like strings, or credential literals.
- Exact normalized duplicates are detected using Unicode NFKC, case-folding, and collapsed whitespace. Stored text is preserved apart from trimming its outer whitespace.
- Because current BASE rows do not carry stable group IDs, non-dialogue rows use the source filename as a provisional group. CHAT-derived dialogue rows use the exact normalized assistant-response hash as their group, preventing the same answer from crossing splits without merging all rows from entire dialogue files. Exact normalized duplicate rows union their groups.
- The deterministic group split targets 80/10/10 by group count, with at least one eligible group in each split. Group sizes can make record-level percentages differ.
- The output is written to `data/processed/base_candidate/`: `train.txt`, `validation.txt`, `test.txt`, `records.jsonl` (exact reviewable records with source line and split), and `manifest.json`. Source-file hashes, split hashes, record-level source locations, group assignments, and duplicate counts are recorded.
- The writer replaces only its four exact output filenames and never deletes unrelated files.
- `.github/workflows/base-candidate-preflight.yml` runs the tests, builds the candidate, and uploads it as `aethrion-base-candidate-preflight`.

## Safety and release state

The generated manifest explicitly keeps `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false`. It does not modify `data/processed/corpus_manifest.json`, any specialist manifest, tokenizer, or checkpoint. On a push to `main`, the workflow stores the unapproved candidate under `candidates/base/{commit_sha}` in `vizyon2027/aethrion-artifacts` only after verifying that the HF repository is private and `HF_TOKEN` is present. No training or model release is triggered.

## Limitations requiring review

- Filename-based grouping for non-dialogue rows is a provisional fallback. A stronger future format should store stable source/group IDs in each BASE row.
- Deduplication only catches exact normalized duplicates, not paraphrases or factual contradictions.
- Structural validation does not establish factual correctness, copyright/license status, privacy clearance, or quality. The script excludes malformed ordinal-month phrases and generic animal-property templates from train/validation/test, caps identical CHAT assistant responses at three examples, and retains every source row with inclusion/exclusion reasons in the review JSONL. These conservative heuristics do not establish semantic quality or approve the remaining records.
- The generated artifact must be inspected and its hashes pinned before any human review decision. This PR does not approve the corpus.
- CHAT/BASE integration remains blocked until the exact candidate snapshot and its source lineage receive the required review.

## Verification status

The PR adds focused tests for duplicate-group union, repeated-answer split leakage, response caps, non-empty splits, SHA-256 consistency, fail-closed flags, and malformed JSONL rejection. On main, the candidate is also retained in the private HF artifact repository under a commit-specific path; it remains explicitly unapproved. The GitHub Actions preflight workflow will produce the first candidate artifact when the PR workflow runs. Until that run succeeds and its artifact is inspected, no record count or successful corpus-preparation claim is made.


## First generated artifact — 2026-11-26

GitHub Actions run [45851315277](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/actions/runs/45851315277) completed successfully. Its artifact is `aethrion-base-candidate-preflight` (artifact ID `15134262131`, archive SHA-256 `b4af5f51a10f549bd72d9660cb9790e2a449b372383ba934f3153f8dd5288b61`).

Verified by downloading and inspecting that exact artifact:

- 93 source JSONL files; 1,792 input rows; 1,792 unique full-text rows; zero exact normalized full-record duplicates.
- 1,006 rows included in candidate splits; 786 retained in the review JSONL but excluded from train/validation/test.
- Exclusions: 722 excess repeated-response rows (response cap = 3), 56 malformed ordinal-month rows, and 36 generic animal-property template rows.
- Candidate split: train 807 rows / 369 groups; validation 98 / 46; test 101 / 46.
- 432 unique included dialogue-response groups; zero identical assistant-response groups crossing splits.
- Review JSONL count and SHA-256 match the manifest; all three split-file SHA-256 values match; every included row's split agrees with its group assignment.
- `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false` remain unchanged.

Split SHA-256:
- train: `b01505a8d9561ec1a665bcb438762068c720cc20744baf255b712ada0073c7cc`
- validation: `88391a79372d72451bc98a86f60959628cc645998e65d007934bfde13328156a`
- test: `baeb6b56916306510e7043c53bc2a31d3a2d6ef006301c02ae524f2fc0e2e92d`

This is a reproducible structural candidate, not an approved training corpus. The 1,242 quality-flag count includes records with overrepresented responses even where the first three examples were retained. Factual review, semantic quality, provenance/rights and full privacy clearance remain open. The pattern scan is a limited pre-screen, not a guarantee that all personal or confidential information has been detected.
