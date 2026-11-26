# BASE candidate preparation — 2026-11-26

## Purpose

Add a reproducible, fail-closed preparation path for the existing BASE JSONL files. The tool creates a candidate snapshot for inspection; it does not replace the disabled legacy approval/preparation flow and does not train a model.

## Implementation

- `scripts/prepare_base_candidate.py` reads every `*.jsonl` file in `data/authored/base/`.
- Each row must be a JSON object with non-empty UTF-8 `text`.
- Exact normalized duplicates are detected using Unicode NFKC, case-folding, and collapsed whitespace. Stored text is preserved apart from trimming its outer whitespace.
- Because current BASE rows do not carry stable group IDs, non-dialogue rows use the source filename as a provisional group. CHAT-derived dialogue rows use the exact normalized assistant-response hash as their group, preventing the same answer from crossing splits without merging all rows from entire dialogue files. Exact normalized duplicate rows union their groups.
- The deterministic group split targets 80/10/10 by group count, with at least one eligible group in each split. Group sizes can make record-level percentages differ.
- The output is written to `data/processed/base_candidate/`: `train.txt`, `validation.txt`, `test.txt`, `records.jsonl` (exact reviewable records with source line and split), and `manifest.json`. Source-file hashes, split hashes, record-level source locations, group assignments, and duplicate counts are recorded.
- The writer replaces only its four exact output filenames and never deletes unrelated files.
- `.github/workflows/base-candidate-preflight.yml` runs the tests, builds the candidate, and uploads it as `aethrion-base-candidate-preflight`.

## Safety and release state

The generated manifest explicitly keeps `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false`. It does not modify `data/processed/corpus_manifest.json`, any specialist manifest, tokenizer, checkpoint, or HF artifact repository. No training or release is triggered.

## Limitations requiring review

- Filename-based grouping for non-dialogue rows is a provisional fallback. A stronger future format should store stable source/group IDs in each BASE row.
- Deduplication only catches exact normalized duplicates, not paraphrases or factual contradictions.
- Structural validation does not establish factual correctness, copyright/license status, privacy clearance, or quality. The script excludes malformed ordinal-month phrases and generic animal-property templates from train/validation/test, caps identical CHAT assistant responses at three examples, and retains every source row with inclusion/exclusion reasons in the review JSONL. These conservative heuristics do not establish semantic quality or approve the remaining records.
- The generated artifact must be inspected and its hashes pinned before any human review decision. This PR does not approve the corpus.
- CHAT/BASE integration remains blocked until the exact candidate snapshot and its source lineage receive the required review.

## Verification status

The PR adds focused tests for duplicate-group union, repeated-answer split leakage, response caps, non-empty splits, SHA-256 consistency, fail-closed flags, and malformed JSONL rejection. The GitHub Actions preflight workflow will produce the first candidate artifact when the PR workflow runs. Until that run succeeds and its artifact is inspected, no record count or successful corpus-preparation claim is made.
