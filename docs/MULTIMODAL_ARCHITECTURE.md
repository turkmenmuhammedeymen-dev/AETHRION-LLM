# AETHRION native multimodal architecture

AETHRION uses a family of separately trained first-party models. No hosted
model APIs, pretrained third-party weights, model proxies, or provider
fallbacks are permitted in production inference.

The model family includes:

- AETHRION-BASE: multilingual chat and reasoning.
- AETHRION-CODE: code generation, explanation, debugging and repair.
- AETHRION-VISION: image understanding and visual question answering.
- AETHRION-IMAGE: text-to-image and image editing.
- AETHRION-ASR: speech recognition.
- AETHRION-TTS: speech generation.
- AETHRION-3D: text/image-to-3D and 3D understanding.
- AETHRION-RESEARCH: local retrieval, ranking and evidence-grounded synthesis.
- AETHRION-DOC: document layout, OCR and semantic understanding.
- AETHRION-SAFE: separately evaluated safety classification and policy support.

Each specialist has its own architecture, processor/tokenizer, training
corpus, training and recovery pipeline, checkpoint, evaluation suite, safety
tests and local inference handler. The shared router selects a specialist but
must never silently substitute another model.

## Runtime contract

A capability is available only when its first-party local checkpoint and all
required companion artifacts exist and pass SHA-256 verification through
aethrion/native_registry.py. Missing or untrained artifacts return an explicit
unavailable status. A capability is not ready merely because its UI tab,
Python class, or API route exists.

The model manifest template is models/manifest.example.json. All models are
initially marked not_trained. A trained status must not be asserted before
dataset approval, completed training, independent evaluation and release
review.

## Data and safety

Use only data with documented rights for the intended training. Speech data
requires speaker consent and suitable privacy controls. Image and 3D datasets
require documented licenses and provenance. Deduplicate and separate training,
validation and held-out test sets. Review privacy, personal data, bias,
copyright, contamination and unsafe content. Keep the training approval gate
closed until these reviews are complete.

Safety requires both trained AETHRION-SAFE components and ordinary runtime
controls: input/output validation, resource limits, access control, audit logs,
abuse testing and human review for consequential actions.

## Current status

The repository currently contains a small from-scratch decoder-only text
Transformer, a byte-level BPE tokenizer, corpus preparation and training
utilities, a local-only model artifact registry, basic document utilities,
and the Gradio interface. The registry and architecture plan do not mean the
specialist models are trained. No production-ready checkpoint for the model
family is claimed. See docs/NATIVE_ONLY_MODEL_FAMILY.md.
