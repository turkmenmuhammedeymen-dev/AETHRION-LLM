# AETHRION-LLM

**Intelligence Without Limits.** AETHRION-LLM is a from-scratch model research
and engineering project. Its production inference policy is local-only:
AETHRION models must be initialized and trained by this project. It does not
load third-party pretrained weights, call Ollama, or silently fall back to
hosted model APIs.

> **Current status:** this repository contains model architectures, data
> governance, training/evaluation scaffolding, checkpoint integrity, and a
> native-only Hugging Face Space. It does not yet contain production training
> corpora or trained model checkpoints. All 11 model entries remain
> `not_trained`; the base corpus manifest remains `training_allowed: false`.
> Passing software tests is not evidence of model capability.

## Model family

AETHRION-BASE, CODE, VISION, IMAGE, ASR, TTS, VOCODER, 3D, RESEARCH, DOC and
SAFE are separate model programs with independent artifacts and release gates.
A feature is unavailable until its own checkpoint, tokenizer/processor,
training report, held-out evaluation, safety review, integrity hashes and
human release approval are present.

See [the native-only model-family specification](docs/NATIVE_ONLY_MODEL_FAMILY.md)
and [the roadmap](docs/ROADMAP.md) for verified implementation status and
remaining work. The cross-product mapping for Vizyon2026/AETHRION, SENTRIX and
TdevKuantum is documented in [the product AI integration matrix](docs/PRODUCT_AI_INTEGRATION_MATRIX.md).
Dataset candidates and their rights-review findings are tracked in
[the dataset source review](docs/DATASET_SOURCE_REVIEW.md).

## Development environment: GitHub Codespaces

Use the repository's `.devcontainer` configuration for remote development,
CPU unit tests, bounded preprocessing and local Space preview. Start with
[the Codespaces runbook](docs/CODESPACES_RUNBOOK.md).

Codespaces is not a GPU training service. The current Hugging Face Space
ZeroGPU runtime is also not a persistent, long-running training environment.
With only those environments, full-scale training of all specialist models is
not currently feasible. Do not represent a smoke test or synthetic fixture as
a trained model.

## Install locally

Python 3.10+ and PyTorch 2.2+ are required.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data,space]"
```

For a CPU-only development environment, CUDA is not required. A compatible
GPU is required for practical large-scale training.

## Data governance and training

Only use material you own, public-domain material, or sources whose terms and
rights explicitly permit the intended machine-learning training use. Preserve
source URLs, licenses, attribution, provenance and SHA-256 hashes. Review
privacy and consent; speech datasets require speaker-consent evidence.
Dataset-level license labels do not automatically clear embedded third-party
web pages, recordings, images, code repositories, or 3D assets.

Keep raw data, prepared corpora, NPZ datasets, tokenizers, checkpoints and
private evaluation material outside Git. The training gate must remain closed
until each dataset and every required source review has been genuinely approved.
Automated checks validate recorded evidence; they do not make legal decisions.

BASE pretraining, instruction tuning, and each specialist training stage have
separate data and evaluation requirements. Follow
[data/specialists/README.md](data/specialists/README.md) for the NPZ schema and
the release gate. Training reports are not releases.

## Tests

```bash
python -m compileall -q aethrion tests hf_space
python -m unittest discover -s tests -v
```

The synthetic smoke-test path is exclusively for software validation. It must
not be used as production training data or as evidence that a model is trained.

## Hugging Face Space

The Space source of truth is tracked in [`hf_space/`](hf_space/). Changes to
that directory are deployed to
[`vizyon2027/AETHRION-LLM`](https://huggingface.co/spaces/vizyon2027/AETHRION-LLM)
by the GitHub Actions deployment workflow after merge to `main`. The workflow
does not change Space hardware and does not upload private training data or
model checkpoints. Keep Space credentials in Space Secrets and GitHub Actions
secrets only; never commit them.

The Space reports a capability as unavailable until the corresponding
first-party model is trained, evaluated, released and installed with verified
artifacts. Deterministic document utilities are not a trained DOC model.

## License and data rights

This repository does not grant rights to third-party training data. Verify
source-specific terms before acquisition, redistribution, or model training.
