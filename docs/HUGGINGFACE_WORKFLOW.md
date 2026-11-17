# AETHRION-LLM — GitHub + Hugging Face workflow

## Source of truth and responsibilities

- **GitHub `turkmenmuhammedeymen-dev/AETHRION-LLM`:** all source code,
  architecture, training scripts, manifests, tests, review history and release
  decisions. Make changes on a feature branch and merge only after CI passes.
- **Hugging Face Space `vizyon2027/AETHRION-LLM`:** deployed Gradio interface.
  The GitHub Actions workflow publishes the tracked `hf_space/` directory
  after approved changes reach `main`.
- **Existing Space ZeroGPU:** short, GPU-decorated application requests and
  hardware diagnostics. It is not a persistent GPU notebook, durable training
  worker, or guaranteed long-running training allocation.
- **Codespaces:** optional fallback for a shell-based task that cannot be
  completed with GitHub's web editor, Actions and the Space.

## Normal development without Codespaces

1. Open the AETHRION-LLM repository on GitHub.
2. Create a feature branch from the latest `main`.
3. Edit files with GitHub's web editor or upload a reviewed source file.
4. Inspect the diff and confirm only intended AETHRION-LLM paths changed.
5. Open a pull request to `main`.
6. Wait for all Python-version tests and the Hugging Face Space runtime smoke
   test to pass.
7. Merge only after reviewing the changed files and CI results.
8. The deployment workflow publishes `hf_space/` to the existing Space.
   Check the Space build/runtime logs and use its **Check ZeroGPU hardware**
   button after deployment.

Do not edit the separate Vizyon2027 product repository as part of this process.
Do not put checkpoints, private data, access tokens or credentials in Git.

## Verify ZeroGPU

The Space's **Check ZeroGPU hardware** button invokes a short
`spaces.GPU`-decorated diagnostic. It reports whether CUDA is visible during
that actual request, the device name and reported memory. It does not load a
model, download data, alter hardware settings or start training.

A ZeroGPU setting in the Space dashboard is not proof that every application
request has a GPU allocated. The diagnostic result is the relevant runtime
observation for that request. GPU availability and quota can vary between
requests.

## Training and checkpoint policy

The current repository has eleven independent model entries. Their
architectures and training/evaluation scaffolding are not trained model
weights. Each capability remains unavailable until its own licensed data,
processor/tokenizer, trained checkpoint, SHA-256 integrity record, training
report, held-out benchmark, safety review and human release approval exist.

The global corpus gate currently remains `training_allowed: false`; all
candidate datasets remain unapproved for training. The project-owner approval
covers public-source review and non-destructive preparation planning only. It
does not grant third-party rights, approve gated datasets, verify speaker
consent, or authorize accepting provider terms. Do not bypass these gates.

ZeroGPU requests are short-lived and may be reclaimed. A training process
must never depend on the Space process filesystem for its only copy of a
checkpoint. Before any approved training run, the project needs a tested
persistent artifact destination, atomic checkpoint writes, optimizer/RNG
resume state, hashes, interruption recovery and an independent evaluation
report. Keep those artifacts outside Git.

Do not launch paid Hugging Face Jobs, change Space hardware, or create paid
storage without a separate explicit compute/budget approval. The current
ZeroGPU allocation is not a basis for promising that all eleven models can be
trained today.

## Current capability truth

The Space interface includes Chat, Code, Vision, Image, 3D, Research, Files and
Audio sections. A visible tab is not a released capability:

- Native chat/code/vision/image/ASR/TTS/3D/research/document/safety models
  remain unavailable until their independent release gates pass.
- Deterministic document creation and text extraction are software utilities,
  not a trained document-understanding model.
- Current 3D research output is a vertex-only point cloud, not a textured,
  watertight mesh.
- The research feature requires a locally approved index; it does not call a
  hosted search provider.
- Uploaded project code is not executed.
- No external model provider fallback is permitted.

## When Codespaces is justified

Use Codespaces only when a specific task needs a persistent interactive shell,
local dependency debugging, a local Space preview, or a bounded preprocessing
operation that is unsuitable for GitHub Actions. Its current dev container is
CPU-only. It is not a substitute for a training GPU.
