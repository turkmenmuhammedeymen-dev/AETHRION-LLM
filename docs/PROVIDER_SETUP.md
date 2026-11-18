# Native-only AETHRION runtime

External inference and search adapters are not part of the production runtime.
Do not configure OpenAI, Anthropic, Gemini, Meshy, Tavily, Ollama, or any other
hosted model/search API for AETHRION. The old provider integration guide has
been retired.

Each capability must use a separate first-party model trained from random
initialization. Model artifacts are registered through
models/manifest.example.json and validated by aethrion/native_registry.py.
Only local, trained, non-empty artifacts with matching SHA-256 hashes can be
resolved. A missing model is unavailable; there is no fallback.

The current model manifest is a template. Every specialist is marked
not_trained. Do not change a model to trained until its architecture,
licensed dataset, processor/tokenizer, checkpoint, evaluation results, and
safety report are all present and reviewed.

See docs/NATIVE_ONLY_MODEL_FAMILY.md for the model-family plan and release
criteria. The existing document extraction/creation utilities are deterministic
software functions and do not require a language-model API.
