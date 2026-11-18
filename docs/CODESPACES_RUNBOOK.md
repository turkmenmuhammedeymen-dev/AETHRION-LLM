# Deprecated: GitHub Codespaces workflow is out of scope

This file is retained only as a historical pointer. **Do not create or use
Codespaces for the current AETHRION-LLM workflow.** The authorized development
surface is the GitHub repository interface, GitHub Actions, and the designated
Hugging Face Space.

The current workflow is documented in
[HUGGINGFACE_WORKFLOW.md](HUGGINGFACE_WORKFLOW.md). The compute boundary is in
[FREE_GPU_TRAINING_RUNBOOK.md](FREE_GPU_TRAINING_RUNBOOK.md). GitHub Actions
provides CPU CI; Hugging Face ZeroGPU is request-scoped and is not a persistent
training worker.

No training job may start until first-party owner-authored data is reviewed
and a suitable compute environment is explicitly authorized.
