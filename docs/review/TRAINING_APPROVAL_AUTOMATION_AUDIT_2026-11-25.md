# Training approval automation safety audit — 2026-11-25

## Critical findings

### 1. BASE preparation could manufacture approval

The previous `scripts/prepare_base.py`:
- contained a hard-coded owner statement and historical session identifier;
- labeled generated/Claude-drafted text as project-owned original text;
- wrote `allow_training: true` into every BASE source entry;
- changed the BASE corpus manifest to `training_allowed: true` and `first_party_origin_attested: true`;
- wrote `approval_status: approved`, rights, source-rights, privacy and quality review booleans as true;
- inserted a fixed reviewer role, timestamp and evidence URI.

The script required an HMAC key, but an HMAC authenticates report bytes; it does not prove the truth of the review claims. A secret-key check cannot replace human review.

### 2. Specialist owner-attestation helper could approve without evidence

The previous `scripts/apply_owner_attestation.py` accepted a session string and statement text, then set:
- governance approval to approved;
- authorship, first-party origin, rights, source-rights, privacy and quality booleans to true;
- specialist manifest training and owner-approval flags to true;
- all source training/privacy flags to true;
- ASR/TTS/VOCODER speaker-consent fields to true solely because the model ID was an audio family.

It did not require a record-by-record review ledger, source-specific rights evidence, privacy evidence, quality evidence, or speaker consent documents. A statement string cannot substantiate these claims.

### 3. CHAT CI runner could call the blanket approval helper

The previous `ci/instruction_tune_chat.py` prepared CHAT data, invoked the blanket owner-attestation helper with a hard-coded session ID and governance statement, then proceeded to instruction tuning and artifact upload. That path could bypass the intended pending-human-review state if its runtime prerequisites were present.

### 4. BASE generator wording overstated review

The previous `data/authored/base/make_base.py` described its generated text as owner-approved and its CHAT material as already attested, although the current BASE corpus manifest is pending and CHAT records have not been individually reviewed.

## Mitigations in this change

- `scripts/prepare_base.py` is now fail-closed and cannot write corpus manifests, sources, or approval flags.
- `scripts/apply_owner_attestation.py` is now disabled and performs no writes.
- `ci/instruction_tune_chat.py` validates the existing CHAT governance record and exits before preparation/training when review is pending; it no longer invokes an automatic attestation helper.
- BASE generator documentation now accurately describes generated candidate text and pending review.
- Regression tests assert that the two deprecated scripts fail without changing governance/manifest files and that the CHAT runner cannot invoke the old helper.

## Current state

No approval flags were opened by this audit. The BASE corpus manifest and all specialist family training gates remain fail-closed. No training was started. This change does not itself complete human review or certify rights, privacy, quality, factuality, or consent.

## Required next steps

1. A responsible human reviewer must inspect the exact source snapshot and retain evidence of the rights basis, permitted training use, provenance, privacy screening, and quality review.
2. Audio families additionally require actual speaker consent evidence; a model-family label is not consent.
3. SAFETY requires separate human safety review.
4. Only then should a controlled, evidence-bound governance workflow be designed and independently tested. No script should infer review completion from a generic owner statement or hard-coded historical session.
5. Rebuild BASE/CHAT artifacts from the reviewed source hashes and rerun held-out evaluation before any training or release decision.

## Scope and limitations

This was a static code-path audit of the four named scripts and their described effects. It did not execute the former approval scripts against production files. The current repository manifests were inspected separately and remain pending/disabled. The mitigations are proposed in this branch and are not active on main until merged.


## Additional provenance wording corrections

The same branch also corrects two generator headers that previously described generated material as first-party/owner-authored: `data/authored/make_expand.py` and `data/authored/research/make_research_more.py`. They now state that the content was generated at the project owner's request and has not been individually human-reviewed. No data records or approval flags were changed.
