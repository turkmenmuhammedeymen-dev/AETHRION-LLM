# Release thresholds (proposed, owner to confirm)

A model may be marked released only when ALL of these hold on a held-out test set it never trained on,
the data was not leaked from its training corpus, and the owner approves in the human-review record.

| Model | Metric | Proposed minimum | Measured 2026-11-27 (CI audit, held-out, synthetic data) |
|---|---|---|---|
| SAFETY | accuracy / macro-F1 on unseen topics | >= 0.95 / >= 0.93, zero BLOCK->ALLOW on the red-team set | 0.884 / 0.872 (red-team BLOCK->ALLOW not measured) |
| CODE | executable pass rate on unseen tasks | >= 0.60 | not measured; 9/11 held-out completions are valid Python, token accuracy 0.932 vs 0.914 baseline |
| RESEARCH | retrieval top-1 on real documents | >= 0.70 | 0.654 top-1, but the untrained lexical baseline scores 0.910, so training currently hurts |
| CHAT | held-out loss and human-rated fluency | loss < 2.5 and >= 80% acceptable replies | 4.12 BASE; 0.040 tuned (leak suspected) |
| VISION/IMAGE/ASR/TTS/VOCODER/3D/DOCUMENT | task metric on real inputs | to be set with real data | synthetic only |

Release always requires: verified checkpoint sha256, provenance, evaluation report, safety review and the
owner's approval record. Passing numbers alone never publish a model.
