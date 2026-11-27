# Dataset governance and technical audit — 2026-11-25

## Purpose and decision

The project owner explicitly confirmed that Claude-generated records have not been individually reviewed. Earlier governance files claimed owner review and approvals for rights, privacy, quality and modality-specific checks. Those attestations were therefore reconciled to pending.

**No training is authorized by this report.** All specialist training flags remain false. This report does not certify legal ownership, provider-term compliance, factual accuracy, consent, or production quality.

## Important correction: source counts vs prepared counts

A first-pass comparison mistakenly treated counts in review sheets and old provenance notes as if they were the raw-source counts. A direct comparison of the actual GitHub JSONL files corrected that conclusion.

For CODE, RESEARCH and SAFETY, the authored `records.jsonl` and prepared manifest `source.jsonl` were fetched and compared record-by-record. They contain identical records in the same order and have matching Git blob identities.

| Family | Authored JSONL rows | Prepared source JSONL rows | Unique group_id values | Review/provenance figures |
|---|---:|---:|---:|---|
| CODE | 163 | 163 | 84 | review sheet 74; provenance says 27 original programs |
| RESEARCH | 694 | 694 | 271 | review sheet 69; provenance says 31 original passages |
| SAFETY | 3,474 | 3,474 | 253 | review sheet 2,110; provenance says 1,894 requests |

Therefore, **there is no current row-count mismatch between these three authored JSONL files and their prepared source copies**. The differences are between current files and older/partial review or provenance figures. The exact transformation history from original seed examples to current expanded records is still not documented sufficiently; that is a lineage/documentation gap, not proof of missing or extra prepared rows.

For other families, the manifest preparation counts and review sheets were found to match at the declared level (VISION 360, IMAGE 360, ASR/TTS/VOCODER 160 each, 3D 210, DOCUMENT 240). Their complete asset-level verification is still pending.

## Direct source-record checks

- CODE: 163 records, 163 unique IDs, 84 group IDs, no exact duplicate `text` values after whitespace/case normalization.
- RESEARCH: 694 records, 694 unique IDs, 271 group IDs, 694 unique queries. The `passage` field has only 278 unique normalized values; 209 passage values are repeated, affecting 625 rows. These are not necessarily duplicate query/passage examples, but repeated passages require review for retrieval-target diversity and split leakage.
- SAFETY: 3,474 records, 3,474 unique IDs, 253 group IDs, no exact duplicate `text` values after whitespace/case normalization.
- For the three families above, authored JSONL and prepared source JSONL matched record-for-record in the retrieved snapshot.

## Governance changes already merged

PR #157 was merged after its GitHub Actions test job and CircleCI build-and-test status passed. It:
- marks all ten specialist manifests as not approved for training;
- clears unverified human-review, rights, privacy, quality and modality-specific attestations;
- marks governance/provenance review as pending;
- keeps CHAT/BASE blocked separately; and
- updates README readiness claims.

Merge commit: `8ee5020faf0be605529767fd2b48633b1a7166e2`.

CI success verifies repository tests and smoke/audit scripts; it does not prove trained-model capability or certify the semantic quality of every data record.

## Family-specific findings

### CODE
- The corpus is a small pilot across Python, JavaScript, SQL, Bash, HTML and CSS, not a broad code corpus.
- Static sample review found issues requiring correction before training: the `Hesap.cek` example accepts a negative withdrawal amount, and the Caesar cipher example only transforms ASCII A-Z/a-z, leaving Turkish letters unchanged. The binary-search example should state its sorted-input precondition.
- The 27-program provenance statement is not a count of current JSONL rows: the current file has 163 rows and 84 group IDs. A reproducible seed-to-variant mapping is required before this expansion can be treated as fully explained.

### VISION / IMAGE
- Generator code builds geometric shape/color/background/position combinations. The data is suitable for limited pipeline tests, not broad visual understanding or general image generation.
- Images are LFS-backed. Generator code and caption records were inspected, but not every binary asset was independently decoded and caption-checked.
- Add per-image decode, dimension, pixel-hash, caption-to-image and train/validation/test leakage checks.

### ASR / TTS / VOCODER
- Generator code constructs formant-like waveforms mathematically; these are synthetic clips, not natural human recordings.
- The pilot does not establish natural Turkish speech recognition, natural-sounding speech synthesis, speaker diversity, prosody, noise robustness or production vocoder quality.
- Verify every waveform's sample rate, duration, finite values, clipping, transcript alignment, feature extraction and group-disjoint split. Speaker-consent status should be represented as not-applicable only if the schema explicitly supports it and synthetic-only provenance is verified.

### 3D
- Generator code creates primitive point-cloud examples. This is not a general 3D asset corpus and does not establish mesh topology, materials, UVs, textures, rigging or real-world object coverage.
- The NPY assets were not all independently decoded and checked in this review. Add per-file shape, finite-value, bounds, point-count, hash and caption/geometry checks.

### RESEARCH
- The current JSONL has 694 rows and 271 group IDs. The provenance note's 31 passages is a stale/seed-level description, not the current row count.
- 694 unique queries map to 278 unique normalized passages; 625 rows share a passage with at least one other row. Determine whether this repetition is intentional multi-query supervision, ensure repeated targets cannot leak across splits, and measure retrieval performance against a genuinely held-out evaluation set.
- Passages have no cited references. Factual accuracy, completeness, currency and source traceability remain unverified. Do not treat this as a verified research knowledge base.

### DOCUMENT
- The data is procedurally generated fictional document/layout content.
- Run a full dataset validator for arithmetic consistency, normalized bounding boxes, token/box/layout/tag alignment, document-type label correctness, rendering and synthetic identity constraints. Existing shape checks alone do not establish visual label alignment.

### SAFETY
- The current JSONL has 3,474 rows, 253 group IDs and 3,474 unique IDs. The older provenance figure (1,894) and review-sheet figure (2,110) are stale or partial relative to the current file; the transformation lineage must be documented.
- Exact duplicate text was not found under whitespace/case normalization, but this does not establish semantic diversity or label correctness. Measure template diversity, category coverage, label distribution, near-duplicates, false positives/negatives and group leakage.
- The policy document is a draft, not a production safety barrier. Independent evaluation and human safety review remain required.

### CHAT / BASE
- The review sheet lists 350 authored prompt/response examples, but the prepared BASE corpus required by the training manifest is absent. Keep CHAT/BASE blocked until the approved corpus, tokenizer, provenance and group-disjoint split checks are complete.
- Check statements about AETHRION, its team and capabilities against the actual product state before using these examples for training.

## Review coverage and limitations

This is a mixed automated and static review, not an exhaustive manual review of every record or binary asset. Text files for CODE, RESEARCH and SAFETY were fetched and parsed for counts, unique IDs, group IDs and exact normalized duplicates. The full contents of every review sheet, every image, waveform, point cloud, NPZ tensor and every semantic label were not all independently reviewed. Legal rights, provider terms, third-party content, privacy, consent and factual correctness cannot be certified by code alone.

## Required remediation before any family can be enabled

1. Produce reproducible seed-to-variant lineage manifests for CODE, RESEARCH and SAFETY; reconcile old review/provenance figures and preserve source hashes.
2. Fix CODE edge cases and add executable tests for negative withdrawals, Turkish alphabet handling and binary-search preconditions.
3. For VISION/IMAGE/3D, run exhaustive asset decoding, integrity, caption/geometry and split-leakage checks.
4. For ASR/TTS/VOCODER, validate all waveform/transcript pairs and modality metadata; report limitations of synthetic-only data.
5. For RESEARCH, resolve repeated passages, enforce group-disjoint splitting, add traceable sources or retain explicit synthetic-draft status, and build a held-out retrieval benchmark.
6. For DOCUMENT, validate arithmetic and all token/box/layout/tag/rendering alignments.
7. For SAFETY, run label and template coverage, near-duplicate, leakage, calibration and independent human error-analysis.
8. For CHAT/BASE, rebuild the missing prepared corpus from approved sources and rerun provenance/preflight checks.
9. Keep all training permissions false until each family independently passes its evidence-backed review and its own task-level evaluation. No automated check should grant legal or human approval.


## Additional full-text record inventory

The following authored JSONL files were also fetched and parsed in the repository snapshot:

| Family | Rows | Unique IDs | Unique group IDs | Primary-text exact duplicate values |
|---|---:|---:|---:|---:|
| VISION | 360 | 360 | 48 | 0 captions |
| IMAGE | 360 | 360 | 48 | 0 captions |
| 3D | 210 | 210 | 210 | 35 repeated descriptions |
| DOCUMENT | 240 | 240 | 81 | 0 text fields |
| CHAT | 1,989 | 1,989 | 441 | 7 repeated prompts |

CHAT has 1,982 unique normalized prompts and 610 unique normalized responses; 210 response values repeat, affecting 1,379 rows. Exact prompt/response pairs are unique. Repeated short responses are not automatically defects, but they may encourage formulaic outputs and should be evaluated for response diversity and factual consistency. The 350 figure in the review sheet is not the current full authored JSONL row count.

For 3D, the 210 rows have 210 distinct group IDs even though the data generator uses a small number of primitive families and preset variations. This means the current grouping may not hold related primitive-family variants together during train/validation/test splitting. Revisit group IDs so that geometrically related examples cannot be split across partitions; the present 210-group count alone does not prove leakage, but it does not adequately demonstrate family-level isolation.

The 35 repeated 3D descriptions are compatible with multiple different point clouds sharing the same short class/preset caption. Caption repetition is not itself a defect; geometry-to-caption alignment and primitive-family split isolation are the key tests.

These counts are source-record metadata checks, not asset-level semantic verification. All image/audio/3D binary objects still require complete decoding and modality-specific integrity checks.
