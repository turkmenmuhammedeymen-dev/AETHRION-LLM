# AETHRION-LLM in GitHub Codespaces

## Purpose and limits

Codespaces is the supported remote development environment for this repository.
It is intended for editing, CPU-based unit tests, data-manifest review, bounded
preprocessing, and local Space/API smoke tests. The included dev container does
not provide a CUDA GPU. Do not use it for long-running or production-scale model
training, and do not assume the Hugging Face Space's ZeroGPU runtime is a
persistent training machine.

The repository and Space remain native-only at inference time: no pretrained
third-party model, Ollama service, or hosted inference API is used. PyTorch,
Python, Gradio, and file-format packages are software infrastructure, not model
providers.

## Create the environment

1. Open the private `turkmenmuhammedeymen-dev/AETHRION-LLM` repository.
2. Select **Code → Codespaces → Create codespace on main**.
3. Wait for `.devcontainer/devcontainer.json` to finish its post-create install.
4. In the Codespaces terminal, verify the environment:

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

CUDA is expected to be unavailable in this CPU development container.

## Validate the repository

Run these checks before changing model code:

```bash
python -m compileall -q aethrion tests hf_space
python -m unittest discover -s tests -v
python -m aethrion.specialist_train --help
python -m aethrion.instruction_tune --help
```

A test-only synthetic smoke test is for software verification only. It is not
training data and must never be used to mark a model trained or released.

## Preview the Hugging Face Space app locally

```bash
cd hf_space
PYTHONPATH=. python app.py
```

Open forwarded port 7860 from the Codespaces **Ports** tab. Without the
Space-only authentication secrets and real, verified model artifacts, native
inference is expected to report unavailable. Do not add model API keys or
third-party inference fallbacks to make the preview appear functional.

Stop the app with Ctrl+C.

## Data and training safety

- Keep raw datasets, private data, NPZ archives, tokenizer outputs, checkpoints,
  and evaluation artifacts out of Git.
- Keep `data/processed/corpus_manifest.json` at `training_allowed: false`
  until a human has verified the exact source rights, intended ML-training use,
  privacy, attribution, provenance, and quality.
- Specialist manifests must independently document dataset/source hashes,
  train/validation/test separation, privacy review, and speaker consent for
  speech data.
- Never approve a dataset based only on a dataset card's top-level license
  when it contains third-party source material with separate rights.
- Never commit access tokens, Space credentials, or other secrets.

Codespaces CPU is not an adequate substitute for a capable training GPU. If the
project is restricted to this repository, Codespaces, and the existing Space
hardware, full-scale training of the entire specialist family is not currently
feasible. Do not start long training jobs or claim model completion under that
constraint.

## Git workflow

Use a feature branch, run the checks above, inspect the diff, and open a pull
request to `main`. The Hugging Face Space deployment workflow publishes only
the tracked `hf_space/` directory after changes reach `main`; it does not
upload training data or checkpoints. Never edit Vizyon2027's separate source
repository as part of this workflow.
