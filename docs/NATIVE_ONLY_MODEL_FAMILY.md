# AETHRION Native-Only Model Family

## Non-negotiable product boundary

AETHRION's production inference must use only model weights trained by this
project from random initialization. Do not download, call, wrap, proxy, distill
from, or silently fall back to a third-party pretrained model or inference API.
No Ollama, OpenAI, Anthropic, Gemini, Hugging Face Inference Providers, Meshy,
Tavily, or equivalent hosted model/search endpoint is part of the native
runtime.

This does not mean the software stack has zero dependencies. Python, PyTorch,
CUDA/ROCm or CPU kernels, file-format libraries, operating systems, and
hardware are execution infrastructure, not pretrained AI models. Training
data must have documented rights that permit the intended training use.
Network access may be used for ordinary source-code collaboration and
documented data acquisition only when separately approved; model inference
must not depend on a remote service.

## Model family (separate weights and training objectives)

| Capability | First-party model to build | Training objective / data |
|---|---|---|
| Chat and multilingual reasoning | AETHRION-BASE | Decoder-only causal LM; licensed multilingual text, instruction tuning, preference/safety evaluation |
| Code | AETHRION-CODE | Separate decoder-only model; permissively licensed source code, tests, explanations, repair pairs |
| Image understanding | AETHRION-VISION | Vision encoder plus language connector / multimodal decoder; licensed image-text pairs and visual QA |
| Image generation | AETHRION-IMAGE | Latent or pixel diffusion/flow model; licensed image-caption pairs, text-image alignment and safety |
| Speech recognition | AETHRION-ASR | Audio encoder and sequence decoder; licensed speech/audio-text pairs |
| Speech generation | AETHRION-TTS | Text/audio acoustic model plus vocoder; licensed speaker-consented recordings |
| 3D understanding/generation | AETHRION-3D | 3D representation encoder/decoder (mesh/point/voxel or hybrid); licensed text/image/3D asset triples |
| Research | AETHRION-RESEARCH | Native retriever/reranker and local index; approved local corpora, citations and provenance; no hosted search API |
| File intelligence | AETHRION-DOC | Document-layout encoder and task model; licensed document/OCR/layout datasets |
| Safety | AETHRION-SAFE | Separate classifier/reviewer plus policy training and adversarial evaluation; not a substitute for runtime controls |

These are independent model programs, not claims that the weights or training
pipelines already exist. AETHRION-BASE may provide language context to
specialists, but it must not be presented as an image, speech, or 3D generator.

## Build gates

1. **Inventory and isolation:** preserve existing working code; create a native
   development branch; inventory every network call, provider adapter, model
   loader, dataset source, secret, and UI claim.
2. **Native runtime contract:** a capability is available only if a local
   first-party checkpoint, matching tokenizer/config, and integrity metadata
   exist. Missing artifacts must produce an explicit unavailable status. No
   fallback is allowed.
3. **Data governance:** each dataset has source provenance, license evidence,
   privacy review, consent where applicable, deduplication, contamination
   checks, and train/validation/test separation. The training approval flag
   remains false until all required evidence is reviewed and approved.
4. **One model at a time:** implement architecture, deterministic smoke test,
   training loop, resume/checkpoint integrity, evaluation suite, and model
   card for each specialist before enabling it in the UI.
5. **Integration:** connect only verified local artifacts; test CPU startup,
   supported GPU inference, resource limits, concurrency, cancellation, and
   failure recovery.
6. **Release gate:** no feature may be labeled ready until CI, model-quality
   evaluations, safety tests, and a live end-to-end test pass. Never equate a
   passing syntax/unit test with model quality.

## Resource and quality reality

A small randomly initialized Transformer is useful for validating a training
pipeline, but it is not competitive with frontier systems. Each specialist
requires substantial licensed data, compute, engineering, and evaluation.
The current repository's small pilot configuration and previously available
Kaggle T4 runtime are development resources only; they cannot establish
frontier-level quality across all modalities. Do not inflate model size beyond
available memory or start expensive/long training without an approved data
manifest and an explicit compute plan.

## Implementation status (verified in the native-specialist branch)

Implemented as randomly initialized PyTorch architectures:

- AETHRION-BASE: causal Transformer + byte-level BPE, pretraining gate and
  resumable checkpoint system already exist.
- AETHRION-CODE: separate model class and independent tokenizer/checkpoint path.
- AETHRION-VISION: image/text contrastive dual encoder.
- AETHRION-IMAGE: text-conditioned pixel diffusion research U-Net.
- AETHRION-ASR: acoustic encoder with CTC objective.
- AETHRION-TTS: mel predictor and duration head; AETHRION-VOCODER is a separate
  learned mel-to-waveform upsampler with its own model ID and training profile.
- AETHRION-3D: text-conditioned point-cloud diffusion; the present output is a
  vertex-only OBJ point cloud, not a watertight textured mesh.
- AETHRION-RESEARCH: dual query/passage encoder for local retrieval training.
- AETHRION-DOC: text/layout encoder with token tagging and extractive-QA heads.
- AETHRION-SAFE: separate ALLOW/REVIEW/BLOCK classifier with reason scores.

The repository now includes a fail-closed multimodal NPZ dataset validator,
separate specialist training runner, response-only AETHRION-BASE instruction
tuning command, example configurations, and forward/loss tests. The audio
preprocessor supports bounded uncompressed PCM WAV and native log-mel features.

Still not complete: high-capacity production architectures, robust document
OCR/layout extraction, a full mesh topology/texture generator, full local
research indexing and evidence citation, an end-to-end trained TTS vocoder
release, inference adapters for every modality, large-scale quality/safety
benchmarks, and all trained checkpoints.

The tracked corpus and every specialist data manifest remain fail-closed.
No production checkpoint or trained specialist release is claimed. Document
creation/extraction remain deterministic utilities, not a trained DOC model.

## Completion definition

A capability is complete only when its first-party model architecture,
licensed dataset manifest, tokenizer/processor, training and resume pipeline,
verified checkpoint, quality benchmark, safety evaluation, local inference
handler, UI integration, and end-to-end tests all exist and pass. Until then,
the UI must label it as research/in development or unavailable, not ready.
