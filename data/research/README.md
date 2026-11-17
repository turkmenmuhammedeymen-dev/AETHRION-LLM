# AETHRION-RESEARCH local index data

Keep approved source files, prepared records and index artifacts local; do not
commit private or restricted material.

1. Copy `approval.template.json` to a private local `approval.json`.
2. Place each reviewed UTF-8 source file beside the manifest or in a safe
   relative subdirectory. Record its exact local path and SHA-256.
3. Prepare `records.jsonl` with one JSON object per record:
   `doc_id`, `source_id`, `title`, and `text`.
4. Record the exact `records_path` and SHA-256 in the approval manifest.
   Any change to the JSONL requires a new human review and updated hash.
5. Keep `index_allowed`, `owner_approval`, `privacy_reviewed`, and each
   source's `allow_index` / `privacy_reviewed` false until rights, privacy,
   provenance and permitted indexing use have actually been reviewed.
6. Build the local index only after all approvals are genuine:

```bash
python -m aethrion.research_index \
  --records data/research/records.jsonl \
  --approval data/research/approval.json \
  --tokenizer artifacts/models/research/tokenizer.json \
  --output-dir artifacts/models/research/index \
  --project-root .
```

The index builder verifies source-file hashes, the exact approved JSONL hash,
source attribution, safe local paths and tokenizer provenance. This does not
prove that a license is valid; a human must review the original license and
privacy/consent requirements. The generated index is not a trained retriever
and does not by itself enable AETHRION-RESEARCH inference.
