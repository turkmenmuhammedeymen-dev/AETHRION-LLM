# AETHRION-LLM API contract (v1)

OpenAI-style, native-only. Requests are routed to `CapabilityRouter`; a model whose
capability has no registered, verified handler is reported unavailable. There is no
hosted-provider fallback and no placeholder answer.

Auth: `Authorization: Bearer <AETHRION_API_TOKEN>` (401 without it, 503 if the server has no token configured).

| Method | Path | Purpose |
| - | - | - |
| GET | `/health` | Liveness (no auth) |
| GET | `/v1/models` | Model ids with `available` and `status` (`ready` / `not_trained`) |
| POST | `/v1/chat/completions` | Text models: `aethrion-chat`, `-code`, `-research`, `-safety` |

Model ids: `aethrion-chat|code|research|document|safety|vision|image|3d|asr|tts`.
Non-text ids are listed by `/v1/models` but use their dedicated endpoints. The `vocoder` entry in the internal training manifest is a TTS component, not a standalone public API model, so it is intentionally excluded from this contract.

Request: `{"model","messages":[{"role":"system|user|assistant","content"}],"max_tokens"?:1-2048,"temperature"?:0-2}`.
`stream` is rejected (`stream_not_supported`). Usage counts are character estimates and labeled `estimated`.

Errors: `{"error":{"message","type","code"}}` — 400 `invalid_request_error`, 404 `model_not_found`,
503 `model_not_trained`, 502 `invalid_model_output`, 500 `model_error`.

Implementation: `aethrion/openai_compat.py` (pure Python, unit-tested) + routes in `aethrion/api.py`.
