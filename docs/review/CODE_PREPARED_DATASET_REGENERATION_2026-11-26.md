# CODE prepared dataset regeneration — 2026-11-26

Status: prepared artifacts regenerated from the corrected authored JSONL. **Training remains disabled.**

## What was done
- Ran `aethrion.specialist_prepare --model code --seed 1` from `data/authored/code/records.jsonl` into a clean directory, twice.
- Both runs produced byte-identical `code.json`, `dataset.npz`, `source.jsonl`, `tokenizer.json` (deterministic).
- Replaced the stale prepared CODE snapshot with the regenerated one.

## Verified
- 163 authored records == 163 prepared source records; ids unique; 84 group ids; text identical.
- `code-0009` (sorted-input contract), `code-0013` (negative/zero withdrawal), `code-0037` (Turkish Caesar) are the corrected versions in the prepared source.
- Split: train 135 / validation 13 / test 15; no group appears in more than one split.
- dataset and tokenizer SHA-256 equal the manifest values.
- 135 records parse as Python; the other 28 are JavaScript/SQL/Bash/HTML/CSS/YAML and are not syntax-checked here.

## Not verified / limits
- Not a semantic review of all snippets, no security audit, no rights/provenance/privacy human review.
- `training_allowed=false`, `owner_approval=false`, `automatic_approval=false`. No approval was created.
- No training was started.
