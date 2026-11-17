# Kaggle workflow retired

AETHRION-LLM no longer uses Kaggle as its documented development or training
workflow. The active remote development environment is GitHub Codespaces.

See [CODESPACES_RUNBOOK.md](CODESPACES_RUNBOOK.md).

Important: Codespaces is configured for development, CPU tests and bounded
data preparation; it is not a GPU training service. The existing Hugging Face
Space ZeroGPU runtime is not a persistent training worker either. Do not use
this historical document to start training or change any dataset approval.

The fail-closed data governance and training contracts remain documented in
[data/specialists/README.md](../data/specialists/README.md) and
[the native-only model-family specification](NATIVE_ONLY_MODEL_FAMILY.md).
