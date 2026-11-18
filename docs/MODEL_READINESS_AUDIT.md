# AETHRION 11-model pre-training readiness audit

Audit scope: repository-tracked manifests and training configurations on the
current development branch. This is a source/configuration audit, not an
executed training preflight; no local checkout or working Python runtime was
available to execute the validators.

## Result

**0 of 11 models are approved to train.** All eleven remain fail-closed.
This is the correct result for the current repository state: no approval
booleans were changed and no training process was started.

| Model | Configuration | Data / approval gate | Result |
|---|---|---|---|
| CHAT | \`configs/small.json\` | Corpus manifest has \`training_allowed: false\`; owner/source attestation and human review are pending; corpus/source evidence and tokenizer provenance are not approved | BLOCKED |
| CODE | \`configs/specialists/code.json\` | Specialist manifest has no sources; training, owner, privacy and human review approvals are false/pending | BLOCKED |
| VISION | \`configs/specialists/vision.json\` | No approved image/text sources or prepared dataset; owner, privacy and independent review pending | BLOCKED |
| IMAGE | \`configs/specialists/image.json\` | No approved image/text sources or prepared dataset; owner, privacy and independent review pending | BLOCKED |
| ASR | \`configs/specialists/asr.json\` | No approved audio sources; speaker-consent and privacy reviews absent | BLOCKED |
| TTS | \`configs/specialists/tts.json\` | No approved speech sources; speaker-consent and privacy reviews absent | BLOCKED |
| VOCODER | \`configs/specialists/vocoder.json\` | No approved audio/mel-waveform sources; speaker-consent and privacy reviews absent | BLOCKED |
| 3D | \`configs/specialists/3d.json\` | No approved point-cloud/conditioning sources or prepared dataset; reviews pending | BLOCKED |
| RESEARCH | \`configs/specialists/research.json\` | No approved first-party research pairs or prepared dataset; reviews pending | BLOCKED |
| DOCUMENT | \`configs/specialists/document.json\` | No approved document/layout sources or prepared dataset; reviews pending | BLOCKED |
| SAFE | \`configs/specialists/safety.json\` | No approved labeled safety dataset; independent safety-label review is absent | BLOCKED |

All specialist manifests currently declare an empty \`sources\` list and
\`training_allowed: false\`. Audio manifests additionally require verified
speaker consent. SAFE requires independently reviewed safety labels. The
vision/image output paths have separate runtime safety gates; dataset approval
alone would not release those capabilities.

## Configuration compatibility checks added

The family preflight now validates each specialist's required architecture
fields, common optimizer/training fields, positive integer dimensions and
finite numeric settings before it can queue a model. It also checks that the
configured vocabulary size is not smaller than the verified tokenizer
vocabulary. The ASR CTC blank token is correctly allowed to be zero.

The existing trainer still performs the full dataset-dependent tensor-shape,
label-range, sequence-length, modality and tokenizer-ID compatibility checks
before allocating the model and optimizer. The family preflight cannot claim
those checks passed while the dataset gate is closed.

## Recovery and report audit

The family supervisor now forwards Ctrl+C/SIGTERM to the active trainer so
its existing signal handler can finish the current step and checkpoint. A
process that could not be spawned is recorded as blocked, not as a started
training run. The session snapshot's completed count includes only terminal
trainer outcomes, not blocked or unstarted models.

The base and specialist trainers retain their checkpoint hash verification,
previous-generation fallback, configuration/data/tokenizer fingerprint checks
and RNG restoration. These are source-reviewed, not runtime-tested in this
audit.

## What must be true before approvals can change

- Each source must be first-party and locally present, with verified ownership
  or rights-holder authority and explicit permission for model training.
- Every source file and prepared dataset must have verified SHA-256 hashes.
- Privacy, quality, provenance, deduplication and group-disjoint train,
  validation and test splits must pass.
- Audio sources require documented speaker consent.
- SAFE labels require independent human review and sufficient held-out
  examples for each class.
- Owner attestation and independent review records must be real, attributable
  and evidence-backed. A code change, chat instruction, empty form or test
  fixture is not evidence of these facts.

After those prerequisites are met, run the default preflight first. Only a
fully successful model-specific preflight can be considered for an explicitly
approved training run. Training completion still does not mean release.
