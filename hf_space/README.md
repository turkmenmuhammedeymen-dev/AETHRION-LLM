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

The chat model remains unavailable until its own trained checkpoint and
matching tokenizer are installed and integrity-verified. Code, vision,
image-generation, ASR, TTS, 3D, research, document-intelligence and safety
specialists are separate first-party model programs and remain unavailable
until each one is independently trained and passes its release gates.

Deterministic document creation and text extraction are available as software
utilities; they are not a trained document-understanding model. Uploaded
project code is not executed. Training is not started from the Space UI.

See the [native-only model-family specification](https://github.com/turkmenmuhammedeymen-dev/AETHRION-LLM/blob/main/docs/NATIVE_ONLY_MODEL_FAMILY.md)
for model-family status and completion criteria. Dataset approval remains
fail-closed until source rights, privacy, provenance and quality checks are
completed.
