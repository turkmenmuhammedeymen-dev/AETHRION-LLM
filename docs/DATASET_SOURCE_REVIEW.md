# Dataset source review — 17 November 2026

This is a source-review register, not a training approval. The project owner has approved proceeding with dataset review. That approval does not replace the dataset creator's license, source-content rights, individual asset rights, privacy rules or speaker consent.

No dataset has been downloaded, hashed, prepared or approved for training by this document. The global corpus gate and all specialist training gates remain fail-closed.

| Model | Candidate source | Publicly declared terms | Review finding | Current decision |
|---|---|---|---|---|
| BASE | [Cosmos Turkish Corpus v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) | Dataset card declares CC BY 4.0; 9,075,453 rows; 22.8 GB download | Corpus aggregates text from third-party websites, forums, news and Wikipedia. Dataset-level CC BY metadata does not independently establish rights for every underlying source item. Attribution and content-level filtering are required. | Candidate only; do not ingest until source-level rights policy and attribution pipeline are accepted |
| ASR | [Mozilla Common Voice Turkish 27.0](https://commonvoice.mozilla.org/en/datasets) | CC0-1.0; Turkish scripted ASR release listed as 2.79 GB | Official terms say contributors dedicate their contributions under CC0 and must provide their own contributions without infringing third-party rights. Verify the exact downloaded release, metadata and consent records. | Best current ASR candidate; prepare only after exact release metadata/hash and consent review |
| TTS / VOCODER | Common Voice Turkish or separately consented speech | Common Voice dataset is CC0 | CC0 does not itself authorize imitating a specific speaker or creating a voice clone. The current manifest requires speaker-consent evidence for TTS/Vocoder. | Blocked until explicit speaker-consent scope and source-specific rights are documented; no voice cloning |
| VISION / IMAGE | [COCO](https://cocodataset.org/#home) | Annotations CC BY 4.0; image files retain their own rights and Flickr terms | COCO states it does not own the image copyrights and users are responsible for compliance with Flickr terms. Annotation license alone does not clear images for model training. | Do not train on image files without per-image rights evidence; annotation-only use requires separate review |
| 3D | [Objaverse](https://huggingface.co/datasets/allenai/objaverse) | Collection ODC-By 1.0; individual assets include CC BY, CC BY-NC, CC BY-SA and CC0 | Individual asset metadata must be read. Exclude NC assets from commercial-intent training. Preserve attribution and source metadata. Share-alike implications require review. | Candidate only; filter by individual asset license and retain attribution before any preparation |
| CODE | The Stack / The Stack v2 | Restricted access was encountered for the queried repositories | Access restriction is not permission. Per-repository and per-file licensing, provenance and source attribution are required. | Do not use unless access is authorized and each source's terms are verified |
| DOC / RESEARCH / SAFE | Official documents, standards and project-owned content | Source-specific | Public availability is not equivalent to permission for ML training. Preserve original URLs, license/evidence, update date and permitted purpose. | No approved corpus yet |

## Source evidence checked

- Cosmos dataset card: https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0
- Common Voice dataset catalogue: https://commonvoice.mozilla.org/en/datasets
- Common Voice legal terms (effective 31 October 2025): https://commonvoice.mozilla.org/en/terms
- COCO terms: https://cocodataset.org/dataset/termsofuse.htm
- Objaverse dataset card: https://huggingface.co/datasets/allenai/objaverse

## Required before opening any training gate

1. Download a pinned release outside Git and record its exact revision, source URL, file list, source license and SHA-256.
2. Keep per-record or per-asset provenance where the source collection mixes licenses.
3. Remove records with unknown rights, personal/sensitive data without a valid basis, opt-out requests, duplicates, corrupted samples and disallowed content.
4. Keep train, validation and test splits source-disjoint to reduce leakage.
5. For ASR, preserve release metadata and speaker consent/provenance; for TTS/Vocoder, obtain explicit consent for the intended speech-generation use and prohibit impersonation.
6. Build the prepared NPZ/JSONL and compute its exact SHA-256.
7. Complete the model-specific approval manifest, privacy review and quality checks. Only then can a human review the gate for that exact prepared dataset.
8. Use a compute plan that fits the available machine. The current Codespaces container is CPU-only and cannot train the full model family at useful scale.

Do not set `training_allowed: true` or mark a model `trained` from a catalog entry, a synthetic fixture, a unit test or a partial preprocessing run.

## Fresh public-catalog checks — 17 November 2026

The following public metadata was checked again on this date. These checks improve source triage; they do not change any training decision.

- **Common Voice Turkish 27.0:** Mozilla Data Collective currently lists the Turkish (tr) scripted-speech ASR release as CC0-1.0 and 2.79 GB. Mozilla's legal terms effective 31 October 2025 describe contributor assurances and the CC0 dedication. This is the strongest current ASR candidate, but the exact downloadable release, metadata, provenance/consent evidence, privacy filtering and file hashes still need to be recorded before preparation. Evidence: https://commonvoice.mozilla.org/lss/datasets and https://commonvoice.mozilla.org/en/terms
- **Cosmos Turkish Corpus v1.0:** the public card currently reports CC-BY-4.0, 9,075,453 rows and 22.8 GB, with URL-attributed content collected from third-party websites, forums, news and Wikipedia. The top-level dataset license does not resolve each underlying source's rights. Evidence: https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0
- **COCO:** the official terms explicitly distinguish CC-BY-4.0 annotations from images whose copyright COCO does not own; image use remains subject to Flickr/source terms. Image training remains blocked without item-level rights evidence. Evidence: https://cocodataset.org/dataset/termsofuse.htm
- **Objaverse:** the dataset card states ODC-By 1.0 for the collection and lists multiple per-object Creative Commons licenses, including non-commercial and share-alike variants. Every asset must be filtered and attributed according to its own metadata. Evidence: https://huggingface.co/datasets/allenai/objaverse
- **The Stack:** BigCode documentation states that each code item remains subject to its original repository license and attribution requirements. The current Hugging Face repository is gated and its access flow requires agreement to terms and sharing contact information with maintainers. No access request or contact-information sharing was made. Evidence: https://www.bigcode-project.org/docs/about/the-stack/ and https://huggingface.co/datasets/bigcode/the-stack

## Owner authorization recorded

The project owner's 17 November 2026 authorization to continue public-source review, evidence recording and non-destructive preparation planning is recorded in [OWNER_DATA_REVIEW_AUTHORIZATION_2026-11-17.md](OWNER_DATA_REVIEW_AUTHORIZATION_2026-11-17.md). This is project authorization only. It does not grant third-party rights, authorize gated access, or approve any data for training.

The dataset candidate register now records the owner-approval scope and the public metadata findings above. All candidate entries remain downloaded=false and training_allowed=false; the global and specialist training gates remain closed.


## Hugging Face mirror verification — Common Voice v27.0 (17 November 2026)

A direct Hugging Face Hub metadata inspection was performed in addition to the official catalogue review:

- `mozilla-foundation/common_voice_17_0` remains discoverable, but its dataset card says that, effective October 2025, Common Voice datasets are exclusively available through Mozilla Data Collective. The Hub Dataset Viewer reports the legacy dataset as empty.
- The expected `mozilla-foundation/common_voice_27_0` repository ID was not found on the Hub.
- The official [Mozilla Data Collective catalogue](https://commonvoice.mozilla.org/lss/datasets) currently lists **Common Voice Scripted Speech 27.0 — Turkish (tr)** as **CC0-1.0**, ASR, MP3, **2.79 GB**. Mozilla's [release announcement](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490) dates v27.0 availability to 18 September 2026.

**Operational decision:** use the official Mozilla Data Collective catalogue as the only current source for discovering the v27.0 Turkish release. Do not build an ingestion job around the legacy Hugging Face dataset ID. This catalogue evidence does not prove the downloaded archive's exact revision, file hashes, speaker-level consent/provenance or privacy status. No download was made and no training permission was granted. The ASR gate remains closed until those exact release checks are recorded and the prepared dataset receives explicit review.
