# Full binary asset audit — 2026-11-26

Tool: `scripts/audit_binary_assets.py` (read-only, exit code 1 on any issue). Every file was checked, not a sample.

| Family | Files/records | Result |
|---|---|---|
| IMAGE | 360 PNG | all decode; all 64x64 RGB; 0 byte-duplicates; captions non-empty |
| VISION | 360 PNG | all decode; all 224x224 RGB; 0 byte-duplicates; captions non-empty |
| ASR | 160 WAV | PCM mono 16 kHz 16-bit; 0.36-1.60 s; 0 silent; transcripts non-empty |
| TTS | 160 WAV | same format; 0 silent |
| VOCODER | 160 WAV | same format; 0 silent |
| 3D | 210 NPY | all float32 (1024,3); no NaN/Inf; coordinates within range; text non-empty |
| DOCUMENT | 240 records | boxes valid (4 values in [0,1]); text non-empty |
| NPZ | 7 datasets | sha256 equals manifest; no NaN/Inf in float arrays |

**Issues found: 0.**

## Limits
- Structural validation only. It does not prove captions/transcripts are semantically correct, that audio is intelligible, or that images/shapes match text beyond what generators define.
- All assets are synthetic pilot data produced by repository generators; no real photographs, voices or documents are included, so no speaker-consent evidence exists or is claimed.
- No approval was created and no training was started.
