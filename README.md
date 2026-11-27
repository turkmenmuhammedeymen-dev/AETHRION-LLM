# AETHRION-LLM

**Intelligence Without Limits.** AETHRION-LLM is a from-scratch model research and engineering project. Its native inference policy is local-only: AETHRION models must be initialized and trained by this project. It does not load third-party pretrained weights or silently fall back to hosted model APIs.

> **Verified status (2026-11-25 audit):** 0 of 11 model-family data preflights are approved for training: the ten specialist families are intentionally blocked pending owner-level record review, rights/provenance/privacy/quality checks and modality-specific review; CHAT/BASE is separately blocked because its prepared corpus/tokenizer and provenance inputs are missing. The specialist datasets are small pilot sets, not production-scale corpora. All 11 registry entries remain `not_trained`; 0 models are release-ready. Passing software tests or a synthetic architecture preflight is not evidence of trained-model capability.

## Model family

AETHRION-BASE, CODE, VISION, IMAGE, ASR, TTS, VOCODER, 3D, RESEARCH, DOC and SAFE are separate model programs with independent artifacts and release gates. A feature remains unavailable until its own checkpoint, tokenizer/processor, training report, held-out evaluation, safety review, integrity hashes and human release approval are present.

See [the native-only model-family specification](docs/NATIVE_ONLY_MODEL_FAMILY.md), [the roadmap](docs/ROADMAP.md), and [the product AI integration matrix](docs/PRODUCT_AI_INTEGRATION_MATRIX.md) for architecture, status and integration scope.

## Source of truth and deployment

GitHub is the source of truth for code, branches, reviews, tests and CI configuration. The Hugging Face Space is the deployed application.

The active GitHub Actions workflow `.github/workflows/aethrion.yml` runs tests for pushes and pull requests to `main`; after a successful push test it stages the canonical runtime and deploys to `vizyon2027/AETHRION-LLM`. The same workflow supports explicit manual audit and training actions. Training is a separate operation and must never be confused with deployment or CI success. The repository also retains `.circleci/config.yml`; do not assume CircleCI jobs are active merely because that configuration file exists.

The GitHub Actions repository secrets used by deployment/training are `HF_TOKEN`, `AETHRION_SPACE_USERNAME` and `AETHRION_SPACE_PASSWORD`; BASE corpus preparation additionally requires `AETHRION_BASE_AUDIT_HMAC_KEY`. The Space separately needs its own `HF_TOKEN` secret for private artifact synchronization. Add secrets only in the appropriate provider settings UI; never commit or paste them into source files or chat.

## Data governance and training

Only use material you own, public-domain material, or sources whose terms and rights explicitly permit the intended machine-learning training use. Preserve source URLs, licenses, attribution, provenance and SHA-256 hashes. Review privacy and consent; speech datasets require speaker-consent evidence. Dataset-level license labels do not automatically clear embedded third-party web pages, recordings, images, code repositories or 3D assets.

Keep raw data, prepared corpora, NPZ datasets, tokenizers, checkpoints and private evaluation material outside Git unless a separately reviewed release policy explicitly permits publication. Each specialist manifest is independently checked for first-party origin, rights, privacy, quality and safety evidence. The current pilot datasets are not approved for training: see [the 2026-11-25 dataset governance reconciliation](docs/review/DATASET_RECONCILIATION_2026-11-25.md). The BASE corpus gate remains closed until its prepared corpus, tokenizer and provenance report are present. Automated checks validate recorded evidence; they do not make legal decisions.

BASE pretraining, instruction tuning and each specialist training stage have separate data and evaluation requirements. Training reports are not releases. Do not represent a smoke test or synthetic fixture as production training data.

## Checkpoints and model artifacts

No model is currently marked trained or released in `models/manifest.json`. Private artifact-repository files and checkpoint filenames alone do not establish that a checkpoint is valid, complete, evaluated or releasable. The deployed Space intentionally excludes unapproved checkpoints. Do not fabricate placeholder weights or change model status until genuine artifacts pass integrity and evaluation gates.

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

Space runtime files are maintained in `hf_space/`. The GitHub Actions deployment assembles the current root `aethrion/` package, support directories and model manifest before deployment. The Space's own README metadata is preserved.

The Space reports a capability as unavailable until the corresponding first-party model is trained, evaluated, released and installed with verified artifacts. Deterministic document utilities are not a trained DOC model.

## License and data rights

This repository does not grant rights to third-party training data. Verify source-specific terms before acquisition, redistribution or model training.
