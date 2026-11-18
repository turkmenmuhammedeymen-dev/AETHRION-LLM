# Deprecated: Kaggle workflow is out of scope

This file is retained only as a historical pointer. **Do not create Kaggle
notebooks, sessions, tokens or training jobs for the current AETHRION-LLM
workflow.** The project scope is limited to the GitHub repository and its
designated Hugging Face Space.

The current compute boundary is documented in
[FREE_GPU_TRAINING_RUNBOOK.md](FREE_GPU_TRAINING_RUNBOOK.md): GitHub Actions
runs CPU tests; Hugging Face ZeroGPU provides short request-scoped inference
and architecture checks. Neither is a persistent training worker.

Real training remains blocked until owner-authored first-party data passes
human review and a suitable compute environment is explicitly authorized.
