# RESEARCH repeated-passage split verification — 2026-11-25

## Follow-up verification

The dataset audit found 694 RESEARCH rows, 694 unique queries, 278 unique normalized passage values, and 209 passage values repeated across multiple rows (625 rows affected).

A follow-up check compared every normalized passage value against its associated `group_id` values:

- 278 distinct normalized passages.
- 209 repeated passage values.
- **0 passage values were associated with more than one group ID.**

The preparation code assigns each record's split from its `group_id` using a deterministic SHA-256-derived bucket. Therefore, repeated copies of the same normalized passage remain in the same split in the current source snapshot. The repeated passages are a target-diversity / corpus-quality concern, but the specific repeated-passage cross-split leakage concern is not supported by this check.

## Remaining concerns

- 694 unique queries still map to only 278 distinct passage texts.
- The factual accuracy, currency, completeness and source traceability of the passages remain unverified.
- Grouping prevents exact repeated-passage values from crossing splits, but does not prove semantic near-duplicate separation or independent evaluation.
- Add explicit regression tests that identical normalized passage values must share one group ID, and that all rows with a group ID remain in exactly one split.

## Status

Documentation-only verification. No source records, prepared artifacts, approval flags, or training settings changed. RESEARCH remains unapproved for training pending human review and task-level retrieval evaluation.
