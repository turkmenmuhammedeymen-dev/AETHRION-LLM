# Binary asset sample validation — 2026-11-25

## Scope

This is a targeted sample-level inspection of the binary assets currently present in the Hugging Face Space repository. It is not a complete audit of all binary files. A full remote batch-validation job could not be started because the Hugging Face Jobs API returned HTTP 402 (Payment Required); therefore no claim is made that all assets were decoded or checked.

## IMAGE and VISION

Inspected four actual PNG payloads from the repository (IMAGE: 0001, 0120, 0360; VISION: 0001, 0360) by fetching their base64 content and decoding/rendering them.

Observed:
- IMAGE sample dimensions are 64×64; VISION samples are 224×224, consistent with the generator's configured resolutions.
- The samples render as clean, simple colored geometric shapes on solid backgrounds, matching their captions in the inspected examples.
- IMAGE examples are extremely small PNG files (roughly 158–226 bytes in the inspected sample); this is consistent with flat-color geometric graphics and is not, by itself, evidence of corruption.
- The generator has 6 shape classes, 8 colors, 3 backgrounds and 5 positions. It selects 360 combinations from the 720 possible combinations for each family, so this is a constrained synthetic pilot rather than broad natural-image coverage.
- The current `group_id` uses shape and color, keeping variations of those combinations together; it does not claim generalization to natural images.

Limit: only four images were decoded and visually inspected. This does not establish that all 720 files/records are valid or correctly captioned.

## ASR / TTS / VOCODER audio

Inspected the WAV headers and sample values for clips 0001, 0080 and 0160, and checked the generator and source JSONL definitions.

Observed:
- Sample clips are valid RIFF/WAVE PCM, mono, 16 kHz, 16-bit.
- Sample durations were 0.94 s, 1.11 s and 1.435 s.
- The three samples had no full-scale clipped samples. Their exact zero-sample fractions were approximately 36.6%, 39.6% and 28.3%, respectively; the generator intentionally inserts silence for spaces and stop consonants, so these values need interpretation alongside a perceptual audio review rather than being treated as automatic failures.
- The source generator creates procedural formant-like signals from phoneme rules; it explicitly does not record or imitate human speakers. There are 160 short phrases and five synthetic voice parameter presets.
- ASR, TTS and VOCODER share the same 160 WAV paths. TTS adds token-duration targets; the generator asserts positive durations and that their sum matches the frame count when generating the corpus.

Limit: only three WAV files were numerically inspected here. No full listening test, intelligibility score, phoneme alignment audit, spectral diversity audit, or all-file decode was completed.

## 3D point clouds

Inspected NPY payloads 0001, 0105 and 0210 directly.

All three:
- Decode as NumPy NPY arrays.
- Have dtype little-endian float32 and shape (1024, 3).
- Contain finite values.
- Have observed coordinate extrema within approximately [-0.95, 0.95].

The generator creates point clouds from seven primitive families and five shape presets. This is a small point-cloud classification/conditioning pilot, not a full mesh pipeline: there are no verified faces, UV maps, materials, textures, skeletons, or animation assets in this source representation.

Limit: only three of 210 NPY files were decoded here. Numeric validity does not prove the point-cloud geometry matches its text label.

## Release decision

All relevant manifests remain fail-closed:
- IMAGE, VISION, ASR, TTS, VOCODER and 3D: `training_allowed=false`, `owner_approval=false`, `privacy_reviewed=false`.
- Source-level `allow_training=false` remains in place.

No binary assets, JSONL records, prepared arrays, tokenizer files, manifests, approval flags, or training settings were changed by this report.

## Required next validation

1. Run a paid/available remote batch job or an owner-controlled local script to decode every PNG, WAV and NPY file.
2. Compare every file path against the corresponding JSONL row and manifest asset hash.
3. For images, validate dimensions, mode, file integrity, pixel diversity, caption-to-shape/color/background/position agreement and split grouping.
4. For audio, validate PCM metadata, duration, clipping, silence ratio, waveform diversity, transcript-to-audio alignment, TTS duration/frame consistency, and perceptual intelligibility with an appropriately qualified human review.
5. For 3D, validate every NPY shape, dtype, finite range, point distribution, geometric family and caption alignment; correct group IDs and rebuild splits before evaluation.
6. Keep training disabled until human rights/provenance/privacy review and task-level evaluation are separately evidenced.
