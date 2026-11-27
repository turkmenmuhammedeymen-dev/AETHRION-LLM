# SAFETY taxonomy and cross-family provenance audit — 2026-11-25

## Scope

Static audit of the current authored JSONL row counts, SAFETY decision/reason-label frequencies, and provenance/generator headers. This is not a semantic review of every row, a legal rights opinion, or an approval to train.

## SAFETY inventory

The current `data/authored/safety/records.jsonl` contains 3,474 rows, 3,474 unique IDs, 253 group IDs, and no empty `text` fields in the checked inventory.

Decision counts:
- ALLOW: 1,128
- REVIEW: 826
- BLOCK: 1,520

Reason-tag counts (multi-label; counts therefore need not sum to the row count):
- 0: 1,128
- 1: 318
- 2: 8
- 3: 320
- 4: 318
- 5: 318
- 6: 316
- 7: 210
- 8: 246
- 9: 766
- 10: 120

## Confirmed taxonomy/provenance discrepancy

The generator `data/authored/safety/make_safety.py` defines reason ID 10 as abusive/profane language. The previous `data/authored/safety/PROVENANCE.md` listed only IDs 0–9. The provenance note was corrected to include ID 10 and to distinguish the current 3,474 generated rows from the older 1,894 inventory figure.

The ID 2 tag occurs only 8 times in the current rows. This is a material coverage warning for the self-harm/support category: aggregate record count does not imply adequate category coverage. The repository's policy requires safe, abstract examples and age-appropriate responses; the low count must be reviewed before any training or release decision.

The decision classes are also imbalanced (ALLOW 1,128 / REVIEW 826 / BLOCK 1,520). Evaluation must disclose per-class metrics and confusion matrices; aggregate accuracy alone is insufficient.

## Cross-family provenance corrections

- CODE generator header previously said "owner-authored", while governance records say Claude generated the material at the owner's request and the owner has not reviewed each record. Header corrected to match the stated workflow.
- SAFETY generator header corrected for the same reason.
- RESEARCH generator header corrected for the same reason. Its current JSONL has 694 rows; the older provenance figure of 31 describes seed passages, not current rows.
- CHAT provenance now states 1,989 current rows and Claude generation at the owner's request rather than asserting direct human authorship.

These are provenance wording corrections only. No source records, prepared NPZ files, tokenizers, manifests, model checkpoints, or training gates were changed.

## Required next work

1. Independently review the SAFETY taxonomy and all reason IDs, especially the sparse ID 2 category and the unbalanced decisions.
2. Add automated checks that the documented reason taxonomy exactly matches the IDs present in the JSONL and generator.
3. Perform semantic review and held-out, per-category safety evaluation with human reviewer agreement.
4. Continue CODE source remediation and rebuild the prepared artifacts through the official preparation pipeline only after the source correction is tested.
5. Keep all 11 family training approvals disabled until rights, provenance, privacy, and modality-specific human review are complete.

## Limitations

The counts above are exact for the fetched current JSONL file and the label values it contains. They do not prove that each label is correct, that examples are diverse, or that any model is safe. No training was started.
