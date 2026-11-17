# AETHRION-LLM capability architecture

## Product boundary

AETHRION-LLM contains a from-scratch decoder-only text model plus separately
registered specialist providers. The base AETHRIONLM is not itself an image,
audio, 3D, search, or code model. Provider-backed capabilities must never be
described as locally trained AETHRION models.

The repository's seven capability groups are the scope of this project:

| UI capability | Router key | Current implementation |
|---|---|---|
| Chat | `chat` | Local AETHRIONLM checkpoint/tokenizer loader; unavailable until a trained checkpoint is supplied |
| Code | `code` | OpenAI-compatible chat-completions adapter; requires provider key and model |
| Image | `image` | OpenAI-compatible image-generation adapter; requires image provider credentials |
| 3D | `3d` | Meshy asynchronous text-to-3D and image-to-3D adapter; currently requests GLB |
| Research | `research` | Tavily search adapter; requires provider credentials |
| File analysis | `file_analysis` | Bounded format-aware parsing plus an explicitly configured analysis provider |
| Audio | `audio` | Turn-based transcription and speech synthesis; not continuous real-time duplex audio |

The Space also has a project-workspace proposal flow. It accepts bounded ZIP
archives, rejects unsafe paths, returns a reviewable proposal ZIP, and does not
execute uploaded code, commit changes, or publish a repository.

## Dispatch contract

`aethrion.routing.CapabilityRouter` is the backend-neutral dispatch boundary.
The UI or API passes the selected capability explicitly. A capability is
available only after its real handler is registered. An unconfigured capability
raises `CapabilityUnavailable`; it must not silently fall back to text chat or
return a fabricated success.

Provider handlers receive a `CapabilityRequest` and return their real result.
Credentials belong in environment variables or a secret manager, never in
source files, committed configuration, logs, or client bundles. Provider
adapters must validate input sizes, timeouts, response types, and output
artifacts. File-writing or repository-editing agents must use an isolated
workspace, inspect diffs, run tests, and require approval before publishing
changes.

## Current implementation and verification status

The repository contains the routing contract, provider adapters, API routes,
Gradio UI, and mocked/unit-level tests. That proves code paths exist; it does
not prove that external providers, the live Hugging Face Space, or a trained
AETHRION checkpoint are operational.

- **Chat:** no trained checkpoint/tokenizer is tracked in the repository. The
  Space correctly reports chat unavailable until matching artifacts are supplied.
- **Code and file analysis:** require `AETHRION_CODE_API_KEY` and
  `AETHRION_CODE_MODEL` (or compatible configured credentials). File analysis
  shares this provider adapter; parsers handle supported document formats.
- **Image:** requires `AETHRION_IMAGE_API_KEY` or `OPENAI_API_KEY`.
- **3D:** requires `MESHY_API_KEY`; task creation and polling are asynchronous.
  Current export request is GLB, not a promise of every 3D format.
- **Research:** requires `TAVILY_API_KEY`.
- **Audio:** requires `AETHRION_AUDIO_API_KEY` or `OPENAI_API_KEY`. It is a
  turn-based speech workflow, not live duplex conversation.
- **Project workspace:** requires the code provider. It returns a review ZIP;
  uploaded code is never executed by the Space.

No live paid-provider integration is claimed as tested unless a separate
opt-in integration test has actually been run. Provider credentials are never
required merely to boot the Space.

## Separate safety model

Safety classification/alignment is a separate planned model and checkpoint,
not a hidden layer of the base language model and not a provider fallback.
See [SAFETY_MODEL_ARCHITECTURE.md](SAFETY_MODEL_ARCHITECTURE.md). It must be
trained and evaluated independently on reviewed, permitted data before it can
be used as a safety control. Until then, deterministic input/output policy
checks and conservative serving limits remain necessary.

## Delivery phases

1. Keep text-model data preparation and training independently fail-closed.
2. Finish checkpoint integrity, resume, interruption recovery, and reproducible
   training tests.
3. Train a small Turkish baseline only after documented dataset/source approval.
4. Add independent held-out evaluation and contamination checks.
5. Build the separate safety classifier and policy-evaluation set; do not mix
   its weights with AETHRIONLM.
6. Complete instruction tuning only with reviewed, licensed examples.
7. Validate each specialist adapter with mocks and opt-in live integration tests.
8. Add authentication, quotas, rate limits, monitoring, and safe public serving.
9. Integrate with the AETHRION Gateway only through a separately reviewed
   adapter and feature flag.

## Claims policy

Measure each capability separately. Do not claim that a provider-backed feature
is an AETHRION-trained model, that a model is safe because a filter exists, or
that AETHRION matches another assistant without reproducible benchmark results.
