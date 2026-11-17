# AETHRION specialist dataset format

All specialist datasets are prepared locally as a NumPy NPZ archive loaded with
`allow_pickle=False`. The archive must contain `sample_id`, `group_id` and
`split` arrays. Every record must be assigned exactly one of `train`,
`validation` or `test`; all three splits must be non-empty, sample IDs must be
unique, and each provenance group must occur in only one split. Text-conditioned
specialist tokenizers are learned from the training partition only and their exact
SHA-256 values are recorded in their dataset manifests.

| Model | Required NPZ arrays |
|---|---|
| BASE instruction tuning | `sample_id, group_id, split, input_ids, labels` |
| Code | `sample_id, group_id, split, token_ids` |
| Vision | `sample_id, group_id, split, images, token_ids` |
| Image | `sample_id, group_id, split, images, token_ids` |
| ASR | `sample_id, group_id, split, features, input_lengths, targets, target_lengths` |
| TTS | `sample_id, group_id, split, token_ids, frame_to_token, target_mels, durations` |
| Vocoder | `sample_id, group_id, split, target_mels, target_waveforms` |
| 3D | `sample_id, group_id, split, points, token_ids` |
| Research | `sample_id, group_id, split, query_ids, passage_ids` |
| Document | `sample_id, group_id, split, token_ids, boxes, layout_ids, tag_labels` |
| Safety | `sample_id, group_id, split, token_ids, decision_labels, reason_labels` |

The manifest is fail-closed. It requires exact boolean approval fields,
reviewed sources with local source paths and matching SHA-256 values, license
and training permission evidence, privacy review, and speaker consent review
for ASR/TTS/Vocoder. Do not set an approval to true unless the responsible human has
actually reviewed and documented that evidence.

Each source entry must contain:
`source_id`, `source_uri`, `source_path`, `sha256`, `license`,
`permitted_use`, `evidence_uri`, `allow_training: true` and
`privacy_reviewed: true`. ASR/TTS/Vocoder also require `speaker_consent_verified: true`. Paths are relative to the manifest folder.


For AETHRION-BASE instruction tuning, use `model_id: "chat"` and
`modality: "text"`. The `input_ids` array contains the prompt and response;
`labels` must be `-100` at prompt positions and contain target token IDs
only at positions the model is expected to learn. The manifest must include
`tokenizer_sha256` matching the tokenizer and verified base checkpoint.
Do not include private chats or personal messages without documented consent
and legal permission.

SAFE manifests must declare `safety_min_per_class` (minimum 25) and provide at least that many ALLOW, REVIEW and BLOCK examples in each of train, validation and test. `reason_labels` are binary multi-hot vectors. This is only a minimum data-quality gate; it does not establish classifier accuracy, calibration, robustness or policy coverage. Do not manufacture harmful operational examples; use reviewed taxonomy labels and benign, prevention-oriented examples.

The TTS model learns transcript-to-mel alignment and durations. The separately
trained Vocoder learns mel-to-waveform conversion; both datasets require
documented speaker consent. TTS cannot be released unless its companion
AETHRION-VOCODER also passes its own evaluation and release gates.

## Raw JSONL preparation (CHAT, CODE, RESEARCH, DOCUMENT, SAFE)

The first-party preparation CLI converts owner-supplied, reviewed JSONL records
into the NPZ contract above, learns byte-BPE tokenizers only from training-split
records for specialist text tasks, copies the original source file for provenance,
and writes SHA-256 hashes. CHAT is the exception: it must reuse the exact BASE
tokenizer from the pretraining checkpoint; the command refuses to create a new
instruction tokenizer that could be incompatible with the BASE weights.

Supported record shapes:

- `chat`: `{"id":"...","group_id":"...","prompt":"...","response":"..."}`; only response tokens are supervised, while prompt and padding positions use the `-100` ignore index. The resulting `model_id: "chat"` manifest can be used by `aethrion.instruction_tune` after its source, privacy and tokenizer reviews are explicitly approved.

- `code`: `{"id":"...","group_id":"...","text":"..."}`
- `research`: `{"id":"...","group_id":"...","query":"...","passage":"..."}`
- `document`: `{"id":"...","group_id":"...","text":"...","boxes":[[x0,y0,x1,y1],...],"layout_ids":[...],"tag_labels":[...]}`; layout arrays must align with byte-BPE token positions.
- `safety`: `{"id":"...","group_id":"...","text":"...","decision":"ALLOW|REVIEW|BLOCK","reasons":[integer,...]}`; decisions are human-supplied, never inferred by the tool.

For CHAT, pass the exact BASE tokenizer generated during BASE pretraining. Do not
train a new tokenizer for the instruction dataset:

```bash
python -m aethrion.specialist_prepare \
  --input data/incoming/chat-reviewed.jsonl \
  --model chat \
  --output-root data/specialists \
  --tokenizer artifacts/tokenizer.json \
  --license "record the exact source license" \
  --permitted-use "record the exact permitted training use" \
  --evidence-uri "https://source.example/license-or-terms" \
  --source-uri "https://source.example/dataset-release"
```

Example (CODE):

```bash
python -m aethrion.specialist_prepare \
  --input data/incoming/code-reviewed.jsonl \
  --model code \
  --output-root data/specialists \
  --license "record the exact source license" \
  --permitted-use "record the exact permitted training use" \
  --evidence-uri "https://source.example/license-or-terms" \\\n  --source-uri "https://source.example/dataset-release"
```

The command always writes `training_allowed=false`, `owner_approval=false`,
`privacy_reviewed=false`, and source-level training/privacy approvals false.
It is a preparation tool, not a license verifier or approval workflow. Do not
change these flags until the actual source terms, creator rights, privacy,
attribution, and any required consent have been reviewed and documented.
Use `group_id` for records that must remain in the same split; the split is
deterministic and group-aware. SAFE requires at least 25 manually reviewed
examples of each decision in every split.

BASE continues to use the separate audited corpus/tokenizer pipeline in `aethrion.prepare_data` and `aethrion.train`. This first JSONL adapter does not yet preprocess raw images, WAV audio, point
clouds, or OCR itself. Those modalities require their own source-specific
decoders and provenance checks; do not relabel text records as multimodal data.

## Raw WAV preparation (ASR, TTS, Vocoder)

The audio adapter `aethrion.specialist_prepare_audio` reads local uncompressed
PCM WAV files, resamples to 16 kHz, extracts native log-mel features, creates
the required padded arrays, and copies each source WAV into the provenance
folder with SHA-256. The JSONL file contains one record per clip:

- ASR: `{"id":"clip-001","speaker_id":"speaker-001","wav_path":"audio/clip-001.wav","transcript":"..."}`
- TTS: the same fields plus `"durations":[...]`, one positive frame count per
  byte-BPE token. Durations must come from a documented, reviewed alignment
  process and must sum exactly to that clip's mel-frame count.
- Vocoder: `{"id":"clip-001","speaker_id":"speaker-001","wav_path":"audio/clip-001.wav"}`

Example:

```bash
python -m aethrion.specialist_prepare_audio \
  --input data/incoming/speech-reviewed.jsonl \
  --model asr \
  --output-root data/specialists \
  --license "record the exact source license" \
  --permitted-use "record the exact permitted training use" \
  --evidence-uri "https://source.example/license-or-terms" \
  --source-uri "https://source.example/dataset-release"
```

For TTS, include the per-token `durations` arrays. For Vocoder, the default
hop length matches the current model config (256 samples); ASR/TTS default to
160. The audio adapter requires either a stable `speaker_id` or explicit `group_id`
on every clip so recordings from the same speaker cannot cross evaluation splits.
The tool never infers speaker consent: all audio source records keep
`speaker_consent_verified=false`, and the dataset manifest keeps
`speaker_consent_reviewed=false`, `privacy_reviewed=false`,
`training_allowed=false` and `owner_approval=false`. A responsible adult
must review the actual license and speaker-consent scope before any approval
is changed. The tool does not download datasets or accept terms.

## Raw image and point-cloud preparation (VISION, IMAGE, 3D)

Install the optional image decoder once:

```bash
python -m pip install -e ".[visual]"
```

VISION and IMAGE JSONL records use `id`, `image_path` (relative to the JSONL
folder), and `caption`. Images are decoded as RGB, bounded, resized to the
exact native model input size (VISION 224px; IMAGE 64px), and stored as uint8
NCHW arrays. The trainer performs the [-1,1] conversion.

3D records use `id`, `point_path` (a local NPY file), and `text`. Point
clouds must already be finite and normalized to [-1,1]; the tool deterministically
samples or repeats points to the configured count (512 by default). This
trains point-cloud generation only, not watertight mesh generation.

Example:

```bash
python -m aethrion.specialist_prepare_visual \
  --input data/incoming/vision-reviewed.jsonl \
  --model vision \
  --output-root data/specialists \
  --license "record exact image rights" \
  --permitted-use "record exact permitted use" \
  --evidence-uri "https://source.example/terms" \
  --source-uri "https://source.example/dataset-release"
```

Per-asset records may override `source_uri`, `license`, `permitted_use`
and `evidence_uri`. This is important for collections such as 3D asset
repositories where each object can carry a different license. The tool still
sets all approval flags false. For DOCUMENT, this tool does not perform OCR:
supply reviewed OCR text, token-aligned normalized boxes and label IDs to the
text/document JSONL adapter. No OCR quality or image rights are inferred.

## Training

Create a per-model manifest by copying `manifest.template.json`, filling the
dataset identity, modality, local NPZ path/hash and reviewed source evidence.
Keep it unapproved until all legal, privacy and consent checks are complete.

Run one specialist, using its matching config:

```bash
python -m aethrion.specialist_train \
  --config configs/specialists/code.json \
  --manifest data/specialists/code/manifest.json
```

Resume only from the same model and same dataset hash:

```bash
python -m aethrion.specialist_train \
  --config configs/specialists/code.json \
  --manifest data/specialists/code/manifest.json \
  --resume artifacts/models/code/checkpoint_last.pt
```

Training produces a checkpoint, SHA-256 sidecar, metrics and a
`training_report.json` marked `NOT_RELEASED`. It does not change the model
registry to `trained`. Independent held-out benchmarks, safety review,
processor/tokenizer integrity and explicit release approval remain mandatory.

These files define the training data contract. No user dataset has been
approved or trained by adding this template.
