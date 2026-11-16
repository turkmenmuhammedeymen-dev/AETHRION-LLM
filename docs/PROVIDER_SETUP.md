# AETHRION provider setup

Only providers with configured credentials are registered. Never commit API keys.

| Capability | Environment variables | Adapter |
|---|---|---|
| Code | `AETHRION_CODE_API_KEY`, `AETHRION_CODE_MODEL`, optional `AETHRION_CODE_BASE_URL` | OpenAI-compatible Chat Completions |
| Image | `AETHRION_IMAGE_API_KEY` or `OPENAI_API_KEY`, optional `AETHRION_IMAGE_MODEL` | OpenAI Images API |
| 3D | `MESHY_API_KEY`, optional `AETHRION_MESHY_MODEL` | Meshy Text-to-3D |
| Research | `TAVILY_API_KEY` | Tavily Search |
| File analysis | Code API key and model | Bounded document parser + chat model |

The code endpoint must implement `POST /chat/completions`. Image generation uses
`POST /images/generations` and requires `data[0].url` or `data[0].b64_json`.
Meshy creates an asynchronous preview task; the returned ID is not a finished
asset. Poll `GET /3d/tasks/{task_id}` for status and output URLs. Research
returns provider-supplied result URLs, not independently verified facts.

Install and start:

```bash
python -m pip install -e ".[api]"
uvicorn aethrion.api:app --host 127.0.0.1 --port 8000
```

Inspect `GET /health` and `GET /capabilities`. Send explicit capability keys
to `POST /dispatch`. File analysis accepts base64 content at
`POST /files/analyze`. Supported types: TXT, MD, JSON, CSV, LOG, PDF and DOCX;
PDF needs the optional `pypdf` package.

The local chat capability appears only when both a trained checkpoint and
tokenizer exist. Remote calls are opt-in; unit tests mock network responses.
Meshy requests may consume paid credits. Keep this API on localhost or behind
authenticated HTTPS: this API does not implement user authentication or rate
limiting and must not be exposed directly to the public internet.
