# Dataset source review — 17 November 2026

This is a source-review register, not a training approval. The project owner authorizes training only on exact datasets whose source-specific rights, provenance, privacy and consent checks pass. That authorization does not replace provider terms or third-party rights.

No candidate listed below has been downloaded, hashed, prepared or approved for training by this document. The global corpus gate and all specialist training gates remain fail-closed.

| Model | Candidate source | Publicly declared terms | Review finding | Current decision |
|---|---|---|---|---|
| BASE | [Cosmos Turkish Corpus v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) | Dataset card declares CC BY 4.0; 9,075,453 rows; 22.8 GB download | Corpus aggregates text from third-party websites, forums, news and Wikipedia. Dataset-level CC BY metadata does not independently establish rights for every underlying source item. | Candidate only; source-level allowlist, attribution and privacy filtering required |
| ASR | [Mozilla Common Voice Turkish 27.0](https://commonvoice.mozilla.org/lss/datasets) | CC0-1.0; Turkish scripted ASR release listed as 2.79 GB | Official catalogue is the current source. Verify exact release metadata, contributor provenance/consent, storage and hashes. | Best current ASR candidate; no download or training until release-specific review |
| TTS / VOCODER | [SpeechDataAI Turkish Conversational Speech](https://huggingface.co/datasets/SpeechDataAI/turkish-conversational-speech) | Provider claims custom commercial license; audio not hosted on HF; paid | Card claims 2,000 hours and consent documentation, but no contract or speaker-consent evidence has been reviewed. No provider contact or purchase was made. | Candidate only; paid acquisition and contract review not authorized; ASR permission does not imply TTS consent |
| TTS / VOCODER | [LJSpeech (Mozilla Data Collective)](https://mozilladatacollective.com/datasets/cmonjhxee01ako007kohpbg34) | CC0-1.0; public-domain text/audio claimed by catalogue | 13,100 WAV clips from one speaker; the catalogue does not itself document the narrator’s explicit consent scope for synthetic voice generation. | Candidate only; exact release, provenance and TTS/Vocoder speaker-consent review required; no download |\n| TTS / VOCODER | [TurkASR-Bench TTS](https://huggingface.co/datasets/yagmurtuncer/turkasr-bench-tts) | Gated; mixed per-engine/source licenses | Synthetic speech benchmark, not real human speech. Some engines/source text have non-commercial or share-alike terms. | Evaluation-only candidate; no gated terms accepted and no production training |
| VISION | [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | Annotations CC BY 4.0; images listed as CC BY 2.0 individually | Google explicitly says it makes no representation or warranty about each image's license and requires per-image verification. It is not a natural-language caption corpus. | Vision candidate only after item-level rights and privacy review; not cleared for IMAGE generation |
| VISION / IMAGE | [COCO](https://cocodataset.org/) | Annotations CC BY 4.0; images subject to Flickr/source terms | COCO says it does not own image copyrights. Annotation license alone does not clear images. | Do not train on image files without per-image rights evidence |
| CODE | [CodeSearchNet](https://github.com/github/CodeSearchNet) | Mixed original repository/file licenses; license records supplied per language | The official repository says source-code licenses are provided in per-language `_licenses.pkl` files. The repository's MIT license covers its tooling, not every sample. | Candidate only; source-level allowlist and provenance required |
| CODE | [The Stack / The Stack v2](https://huggingface.co/datasets/bigcode/the-stack) | Mixed/source-specific; gated access | Access restrictions and provider terms remain unresolved. | Do not request access or accept terms; use only an independently authorized, per-file-cleared release |
| DOCUMENT | [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet) | Dataset card declares CDLA-Permissive-1.0; 80,863 annotated pages | Strong layout-supervision candidate. Upstream document provenance/privacy still needs review, and current AETHRION-DOC needs OCR text paired with layout boxes. | Candidate only; privacy filtering, source provenance and OCR schema work required |
| RESEARCH | [MS MARCO](https://microsoft.github.io/msmarco/Datasets.html) | Non-commercial research terms; no underlying IP rights granted | Microsoft says use is non-commercial research only and warns it may not own underlying document rights. | Do not use for product training under current intended-use assumptions |
| 3D | [Objaverse](https://huggingface.co/datasets/allenai/objaverse) | Collection ODC-By 1.0; per-asset CC BY, CC BY-NC, CC BY-SA and CC0 | Per-asset metadata is authoritative. Exclude NC assets for product use and review share-alike obligations; retain attribution. | Candidate only; select only individually cleared assets |
| 3D | [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive) | Non-commercial research and educational use only | Official terms restrict use to non-commercial research/education and require gated agreement. | Exclude from product training; no access or terms accepted |
| ASR evaluation | [TurkMedSTT Turkish ASR benchmark](https://huggingface.co/datasets/turkmedstt/turkish-asr-benchmark) | CC BY 4.0 benchmark metadata | Metrics only; source audio and transcripts are not redistributed. | Evaluation metadata only; not a training corpus |
| SAFE | Original AETHRION-authored corpus | No external license; corpus not yet created | Must be authored and independently human-reviewed using the ALLOW/REVIEW/BLOCK taxonomy, without operational harmful instructions or private conversations. | No data exists yet; authoring and review required |

## Additional source findings

- Open Images V7: annotations are CC BY 4.0; images are listed as CC BY 2.0, but Google explicitly disclaims guaranteeing image license status. Verify every selected image. https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- DocLayNet: HF dataset card declares CDLA-Permissive-1.0 and reports 80,863 human-annotated pages. https://huggingface.co/datasets/docling-project/DocLayNet
- CodeSearchNet: source licenses are supplied per language in `_licenses.pkl`; do not infer sample licensing from the repository's MIT license. https://github.com/github/CodeSearchNet
- MS MARCO: terms restrict use to non-commercial research and do not extend underlying IP rights. https://microsoft.github.io/msmarco/Datasets.html
- Objaverse: collection ODC-By 1.0 with per-object CC licenses; the object-level metadata must be filtered. https://huggingface.co/datasets/allenai/objaverse
- LJSpeech: Mozilla Data Collective lists CC0-1.0, 13,100 single-speaker WAV clips and public-domain text/audio, but explicit narrator consent scope for voice synthesis still needs review. https://mozilladatacollective.com/datasets/cmonjhxee01ako007kohpbg34\n- SpeechDataAI: the HF card describes a paid commercial Turkish speech dataset, but the audio is not hosted and the contract/consent evidence has not been reviewed. https://huggingface.co/datasets/SpeechDataAI/turkish-conversational-speech
- TurkASR-Bench TTS: synthetic Turkish audio, gated, with mixed per-engine/source licenses; not a general production TTS corpus. https://huggingface.co/datasets/yagmurtuncer/turkasr-bench-tts
- ShapeNetCore: non-commercial research/education restriction; excluded from product training. https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive

## Required before opening any training gate

1. Pin the exact release/revision and record its file inventory, source URI, declared license and SHA-256.
2. Preserve per-record or per-asset provenance for mixed-license collections.
3. Exclude records with unknown/incompatible rights, sensitive personal data without a valid basis, opt-out requests, duplicates, corrupted samples and disallowed content.
4. Keep train, validation and test splits source-disjoint to reduce leakage.
5. For ASR, preserve contributor consent/provenance; for TTS/Vocoder, obtain explicit consent for the intended speech-synthesis use.
6. Build the prepared NPZ/JSONL, validate modality-specific shapes, and compute its exact SHA-256.
7. Complete the model-specific approval manifest and privacy/quality review. Only then can the training gate be opened for that exact prepared dataset.
8. Keep paid datasets, provider contracts, gated click-through terms and paid compute outside this authorization.

Do not set `training_allowed: true` or mark a model `trained` from a catalog entry, a synthetic fixture, a unit test or a partial training run.
