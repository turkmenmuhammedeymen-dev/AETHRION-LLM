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

## Primary workflow: GitHub + Hugging Face

GitHub is the source of truth for code, branches, pull requests, automated
tests and the Space deployment workflow. The Hugging Face Space is the live
application and its existing ZeroGPU allocation is used only for bounded,
GPU-decorated inference/diagnostic requests. See the
[Hugging Face-first workflow](docs/HUGGINGFACE_WORKFLOW.md).

Codespaces is optional: use it only when a local shell, dependency debugging
or a task that cannot be completed through GitHub's repository interface and
CI is genuinely needed. It is CPU-only in this project.

ZeroGPU is not a persistent training machine and does not make a full model
family trainable by itself. Do not start paid HF Jobs or change Space hardware
without a separately approved compute plan. Do not represent a smoke test or
synthetic fixture as a trained model.

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

AETHRION's current data policy permits only first-party, owner-authored
material. Do not download, prepare, or train on third-party datasets, including
public-domain or openly licensed corpora. For each owner-authored source,
preserve documented authorship, rights-holder role, provenance and SHA-256
hashes. Review privacy and quality; speech data additionally requires explicit
speaker consent for the intended model objective. Automated checks validate
recorded evidence but do not make legal or consent decisions.

Keep raw data, prepared corpora, NPZ datasets, tokenizers, checkpoints and
private evaluation material outside Git. The training gate must remain closed
until each dataset and every required source review has been genuinely approved.

Specialist test fixtures are explicitly marked test-only and are validated in
schema/integrity mode without training approval; production preflight continues
to require the full source-rights and human-approval checks.
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
