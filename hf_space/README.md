---
title: AETHRION-LLM
emoji: ✨
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# AETHRION-LLM
Intelligence Without Limits.

The Space interface groups Chat, Code, Vision, Image, 3D, Research, Files and
Audio. The production runtime is native-only: no third-party inference,
image, speech, 3D, code or hosted-search APIs are called.

Set AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD under Space Secrets
to enable authenticated access to local model inference. No model API keys
are accepted or required.

The Space provides a public, read-only 11-model training-readiness panel. It reports non-sensitive data-gate blockers without requiring inference authentication; it never starts training or writes checkpoints. A READY result is not a trained or released model.\n\nThe chat model remains unavailable until its own trained checkpoint and
matching tokenizer are installed and integrity-verified. Code, vision,
image-generation, ASR, TTS, 3D, research, document-intelligence and safety
specialists are separate first-party model programs and remain unavailable
until each one is independently trained and passes its release gates.

The current VISION architecture provides image/text embedding similarity;
it is not OCR, image captioning, or visual question answering. The current DOC
adapter extracts existing text and uses placeholder page geometry; it is not
an OCR engine. AETHRION-3D generates vertex-only point clouds, not textured
watertight meshes. Image-to-3D conditioning and video generation are not
implemented. Image and 3D file-producing capabilities remain unregistered in the
local router until a dedicated visual/geometry output safety verifier is
implemented, tested and release-gated. Text SAFE alone cannot review those files.

Deterministic document creation and text extraction are available as software
utilities; they are not a trained document-understanding model. Uploaded
project code is not executed. Training is not started from the Space UI.

Cross-product integration plans for Vizyon2026/AETHRION, SENTRIX and
TdevKuantum are tracked in the repository's
[native product AI integration matrix](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/docs/PRODUCT_AI_INTEGRATION_MATRIX.md).
Those legacy product endpoints are not yet connected to released native
checkpoints; no external provider fallback is enabled.

See the [native-only model-family specification](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/docs/NATIVE_ONLY_MODEL_FAMILY.md)
for model-family status and completion criteria. Dataset approval remains
fail-closed until source rights, privacy, provenance and quality checks are
completed.

## Current status (2026-11-26)

All eleven model registry entries remain **NOT_TRAINED and NOT_RELEASED**. The ten specialist-family manifests are fail-closed: `training_allowed=false`, `owner_approval=false` and `privacy_reviewed=false`, pending documented human review. CHAT/BASE is separately blocked because its prepared corpus, tokenizer and provenance inputs are missing. The prepared specialist datasets are small synthetic pilot candidates drafted/generated for this project; they are not approved training corpora and their record counts are not production-scale evidence.

The latest private-artifact evaluation on 2026-11-26 found that the CODE checkpoint does not match the current dataset manifest, so the integrity/evaluation audit exited with failure. Other checkpoint evaluation outputs are synthetic held-out measurements only, not a release decision. Private artifact files do not establish training approval or release readiness. The Space deployment includes the current code and dataset manifests, but unapproved model checkpoints are intentionally removed from the Space. No model is approved for release.

The metrics recorded in `docs/RELEASE_THRESHOLDS.md` are pilot measurements, not proof of production capability. SAFETY and RESEARCH are below their proposed thresholds; CODE's next-token accuracy is not executable code-task success. Real-world evaluations are missing for several modalities. A synthetic preflight, passing CI, a checkpoint filename or a successful training session does not establish model quality or release readiness.

## Private artifact and training setup

The owner-only panel can run one bounded, resumable, data-gate-checked training session only after the relevant family passes the documented approval gate. It does not bypass dataset approvals or release gates. For private checkpoint synchronization, configure the Space secret `HF_TOKEN` with write access to the private `vizyon2027/aethrion-artifacts` model repository and the Space variable `AETHRION_ARTIFACT_REPO=vizyon2027/aethrion-artifacts`. Add secrets only in Hugging Face Space Settings → Variables and secrets. The GitHub Actions workflow separately requires its own repository secrets. A healthy Gradio endpoint does not prove that private artifact synchronization is configured.

The chat model remains unavailable until its own trained checkpoint and matching tokenizer are installed and integrity-verified. Code, vision, image-generation, ASR, TTS, 3D, research, document-intelligence and safety specialists are separate first-party model programs and remain unavailable until each one is independently trained and passes its release gates.
