# AETHRION-LLM

**Intelligence Without Limits.** AETHRION-LLM is a from-scratch model research and engineering project. Its native inference policy is local-only: AETHRION models must be initialized and trained by this project. It does not load third-party pretrained weights or silently fall back to hosted model APIs.

> **Verified status:** the repository contains model architectures, data governance, training and evaluation scaffolding, checkpoint-integrity code, and a Hugging Face Space application. It does not currently contain production training corpora or trained model checkpoints. All 11 native model-family entries remain `not_trained`; the base corpus manifest keeps `training_allowed: false`. Passing software tests is not evidence of model capability.

## Model family

AETHRION-BASE, CODE, VISION, IMAGE, ASR, TTS, VOCODER, 3D, RESEARCH, DOC and SAFE are separate model programs with independent artifacts and release gates. A feature remains unavailable until its own checkpoint, tokenizer/processor, training report, held-out evaluation, safety review, integrity hashes and human release approval are present.

See [the native-only model-family specification](docs/NATIVE_ONLY_MODEL_FAMILY.md), [the roadmap](docs/ROADMAP.md), and [the product AI integration matrix](docs/PRODUCT_AI_INTEGRATION_MATRIX.md) for architecture, status and integration scope.

## Source of truth and deployment

GitHub is the source of truth for code, branches, reviews, tests and CI configuration. The Hugging Face Space is the deployed application.

The CircleCI workflow in `.circleci/config.yml` runs:
1. `build-and-test`: installs the project from `pyproject.toml` and runs `python -m pytest -q`.
2. `deploy-to-hugging-face`: after tests pass, stages the current AETHRION package and approved support files into `hf_space/`, then uploads that folder to `vizyon2027/AETHRION-LLM`.

The deployment is enabled for `main` and `circleci-project-setup`. It preserves the Hugging Face Space's root `README.md` because that file contains Space metadata. The repository README is also staged as `docs/REPOSITORY_README.md` in the Space. GitHub-only workflow and development-container files remain in GitHub; they are not runtime dependencies of the Space.

The CircleCI context `aethrion-deploy` supplies `HF_TOKEN`. Keep credentials in CircleCI contexts and Hugging Face Space Secrets; never commit tokens, API keys or private credentials.

## Data governance and training

Only use material you own, public-domain material, or sources whose terms and rights explicitly permit the intended machine-learning training use. Preserve source URLs, licenses, attribution, provenance and SHA-256 hashes. Review privacy and consent; speech datasets require speaker-consent evidence. Dataset-level license labels do not automatically clear embedded third-party web pages, recordings, images, code repositories or 3D assets.

Keep raw data, prepared corpora, NPZ datasets, tokenizers, checkpoints and private evaluation material outside Git unless a separately reviewed release policy explicitly permits publication. The training gate must remain closed until the required first-party origin, rights, privacy, quality and safety reviews have been completed. Automated checks validate recorded evidence; they do not make legal decisions.

BASE pretraining, instruction tuning and each specialist training stage have separate data and evaluation requirements. Training reports are not releases. Do not represent a smoke test or synthetic fixture as production training data.

## Checkpoints and model artifacts

No trained checkpoint or production model weights are currently present in the tracked repository or deployed Space. The paths in `models/manifest.json` describe expected outputs; they are not evidence that those files exist. The manifest correctly marks each model `not_trained`. Do not fabricate placeholder weights or change model status until genuine artifacts pass integrity and evaluation gates.

## Development and tests

Python 3.10+ and PyTorch 2.2+ are required for local development. A compatible GPU is needed for practical large-scale training.

Install:
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[api,visual,documents,space,data]"
```

Run tests:
```bash
python -m pytest -q
```

Codespaces is optional and CPU-only in this project. Hugging Face ZeroGPU is not a persistent training machine and does not make a full model family trainable by itself. Do not start paid HF Jobs or change Space hardware without a separately approved compute plan.

## Hugging Face Space

Space runtime files are maintained in `hf_space/`. CircleCI stages the current root `aethrion/` package and the project support directories (`configs/`, `docs/`, `data/`, `tests/`, `ci/`, `scripts/`), plus `pyproject.toml` and the example model manifest, before deployment. The Space's own README metadata is preserved.

The Space reports a capability as unavailable until the corresponding first-party model is trained, evaluated, released and installed with verified artifacts. Deterministic document utilities are not a trained DOC model.

## License and data rights

This repository does not grant rights to third-party training data. Verify source-specific terms before acquisition, redistribution or model training.
