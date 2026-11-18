# Kaggle optional free-GPU workflow

Kaggle is optional. It is not required for editing the repository or running
GitHub Actions tests. Use it only when an approved training dataset exists and
a free GPU is actually offered.

See [FREE_GPU_TRAINING_RUNBOOK.md](FREE_GPU_TRAINING_RUNBOOK.md) for the
current safe workflow, checkpoint handling and fail-closed data gates.

Kaggle's free GPU availability, queueing, hardware, session limits and weekly
quota can change. Do not purchase compute or use billing-linked services.
GitHub Actions remains the CI environment; Hugging Face ZeroGPU is not a
persistent training worker.

Do not train until the exact source license, privacy and (for audio) speaker
consent have been reviewed and the model-specific manifest passes validation.
