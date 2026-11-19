# AETHRION-LLM — GitHub + Hugging Face workflow

## Source of truth and responsibilities

- **GitHub `turkmenmuhammedeymen-dev/AETHRION-LLM`:** source of truth for application source, model architectures, training code, manifests, tests, review history and release decisions.
- **CircleCI `.circleci/config.yml`:** the active CI/deployment pipeline. It runs `build-and-test` and, only after that job succeeds, `deploy-to-hugging-face`.
- **Hugging Face Space `vizyon2027/AETHRION-LLM`:** deployed Gradio application and runtime environment. Its root `README.md` contains required Space metadata and is intentionally preserved by deployment.
- **Kaggle/Colab free GPU:** the intended user-operated route for actual training, subject to free-tier availability and session limits. A Space ZeroGPU request is not a persistent training worker.
- **Codespaces:** optional development shell; not a training GPU.

## What the current CircleCI pipeline does

On `main` and `circleci-project-setup`:

1. `build-and-test` checks out the repository, installs `.[api,visual,documents,space,data]` from `pyproject.toml` plus pytest, and runs `python -m pytest -q`.
2. The same job then runs `python ci/space_smoke.py` to import the Gradio Space and exercise its safe no-secret runtime/document paths.
3. `deploy-to-hugging-face` runs only after both test commands succeed.
3. The deployment stages the current root `aethrion/` package and the `configs/`, `docs/`, `tests/`, `data/`, `ci/` and `scripts/` directories into `hf_space/`.
4. It also stages `pyproject.toml` and `models/manifest.example.json`, copies the repository README to `docs/REPOSITORY_README.md`, removes the private owner-review authorization document, and uploads `hf_space/` to the Space.

The pipeline deliberately does **not** copy GitHub's `.circleci/` configuration into the Space: it is CI configuration, not a Space runtime dependency. Likewise, the Space root README is preserved because it contains the Gradio SDK metadata. The repository README is available inside the Space at `docs/REPOSITORY_README.md`.

The CircleCI context `aethrion-deploy` supplies `HF_TOKEN`. Keep credentials in CircleCI contexts and Hugging Face Space Secrets. Never commit tokens, passwords, API keys or private credentials.

## Normal development and release

1. Start from the latest GitHub `main` and make changes on a feature branch.
2. Review the complete diff; keep changes limited to intended AETHRION-LLM files.
3. Open a pull request to `main`.
4. Require the CircleCI `build-and-test` job to pass before merging.
5. After merge, confirm the `deploy-to-hugging-face` job succeeds.
6. Verify the resulting files in `vizyon2027/AETHRION-LLM` and inspect the Space build/runtime logs.

A green `build-and-test` job proves the unit tests and the Space runtime smoke test passed in CircleCI; it does not prove the live Space is currently healthy, that a GPU was allocated, or that any model has been trained. The smoke test imports the Gradio app and exercises selected no-secret runtime paths; it is not a live Hugging Face deployment health check.

Do not edit the separate Vizyon2027 product website/repository as part of this process. The Hugging Face Space named above is the AETHRION-LLM deployment target.

## ZeroGPU diagnostics

The Space's **Check ZeroGPU hardware** button invokes a short GPU-decorated diagnostic. It reports whether CUDA is visible during that request, the device name and reported memory. It does not load a model, download a dataset, change hardware settings or start training.

The **Run 11-model ZeroGPU preflight** button exercises model architectures with disposable synthetic tensors. It is not a training run: it performs no optimizer step, writes no checkpoint and does not create production weights. GPU availability and quota can vary between requests.

## Training data and model release gates

The repository currently defines eleven independent native model entries. Their architectures and training/evaluation scaffolding are not trained weights. The verified manifests currently mark all eleven entries `not_trained`; the base corpus manifest keeps `training_allowed: false` and `first_party_origin_attested: false`.

Training requires real, appropriately authorized data and completed provenance, rights, privacy, quality, deduplication, split-integrity and relevant consent reviews. Audio datasets require speaker consent; SAFE data requires separate human safety review. Do not treat blank templates, generated examples, synthetic preflight tensors or candidate-source lists as an approved production corpus.

Each model release requires its own genuine checkpoint, matching tokenizer/processor, SHA-256 integrity record, training log, held-out evaluation, safety review and human release approval. Do not change a model status or open the training gate without the corresponding evidence.

## Checkpoints, storage and compute

ZeroGPU requests are short-lived and may be reclaimed. Never rely on the Space process filesystem as the only copy of a checkpoint. Approved training needs tested persistent artifact storage, atomic checkpoint writes, optimizer/RNG resume state, interruption recovery, hashes and independent evaluation reports. Keep raw/private data and training artifacts outside Git unless a separately reviewed release policy explicitly permits publication.

Use free Kaggle/Colab GPU sessions where available, and train one model at a time. Do not launch paid Hugging Face Jobs, change Space hardware, or create paid storage without separate explicit compute/budget approval. The current Space ZeroGPU allocation is not a persistent or guaranteed training environment.

## Current capability truth

A visible Space tab is not proof of a released model capability:

- Native chat, code, vision, image, ASR, TTS, 3D, research, document and safety models remain unavailable until their independent release gates pass.
- Deterministic document creation and text extraction are software utilities, not a trained document-understanding model.
- Current vision architecture provides image/text embedding similarity; it is not OCR, image captioning or visual question answering.
- Current 3D research output is a vertex-only point cloud, not a textured watertight mesh. Image-to-3D and video generation are not implemented.
- The research feature requires a locally approved index; it does not call a hosted search provider.
- Uploaded project code is not executed.
- No external model-provider fallback is permitted.

## Security and scope

Do not put checkpoints, private datasets, personal information, tokens or credentials in Git or public Space files. Do not bypass source-rights, privacy, consent, safety or release gates. Automated checks validate recorded evidence; they do not make legal decisions.
