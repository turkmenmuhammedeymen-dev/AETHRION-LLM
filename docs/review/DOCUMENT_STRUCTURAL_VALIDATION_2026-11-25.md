# DOCUMENT structural validation — 2026-11-25

## Scope

A complete pass was performed over all 240 authored DOCUMENT JSONL records in `data/authored/document/records.jsonl`. This is a structural and rule-based check of the text/annotation records, not a visual inspection of every rendered page or a proof of semantic tag correctness.

## Inventory

- 240 records total.
- 80 invoices, 80 application forms, 80 dispatch notes.
- Text lengths: 118–201 Unicode characters.
- Annotation lengths: 15–41 token-level entries per record.

## Checks performed

Across all 240 records:
- JSONL records parse successfully.
- Every record has non-empty text.
- `boxes`, `layout_ids`, and `tag_labels` have equal lengths.
- Every bounding box has four finite coordinates in [0,1], with x_min ≤ x_max and y_min ≤ y_max.
- Layout IDs are integers in the configured 0–6 range.
- Tag labels are integers in the configured 0–7 range.
- All 80 invoice totals equal the sum of their line-item amounts; no arithmetic mismatch was found.

## Important limits and open risks

- The source generator creates fictional documents from fixed synthetic name, city, company and product lists; this is a small templated pilot, not broad document understanding coverage.
- The checks above do not prove each token's tag or bounding box matches the exact visible word, nor that all generated page layouts render as intended.
- No OCR-style image rendering is present in this corpus; it is structured text with token-aligned boxes/layout/tag attributes.
- The prepared NPZ was not decoded in this pass. The binary batch job was unavailable (HF Jobs API returned HTTP 402), so prepared-array integrity and source-to-NPZ equivalence remain unverified.
- Rights/provenance review and independent human semantic review remain pending.

## Release decision

The DOCUMENT manifest remains fail-closed: `training_allowed=false`, `owner_approval=false`, and `privacy_reviewed=false`. No source rows, NPZ artifacts, tokenizer, manifest, approvals, or training settings were changed.
