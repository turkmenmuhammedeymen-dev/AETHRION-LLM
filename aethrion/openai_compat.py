"""OpenAI-style request/response contract for the native AETHRION-LLM API.

Pure Python (no torch/fastapi): it only validates, maps and shapes data. The
real model work stays in ``CapabilityRouter`` handlers. A model id whose
capability has no registered handler is reported as ``not_trained`` /
``unavailable``; this module never invents an answer.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

MODEL_IDS: dict[str, str] = {
    "aethrion-chat": "chat",
    "aethrion-code": "code",
    "aethrion-research": "research",
    "aethrion-document": "document",
    "aethrion-safety": "safety",
    "aethrion-vision": "vision",
    "aethrion-image": "image",
    "aethrion-3d": "3d",
    "aethrion-asr": "asr",
    "aethrion-tts": "tts",
}
# Only text-in/text-out capabilities can be served by /v1/chat/completions.
CHAT_COMPLETION_CAPABILITIES = frozenset({"chat", "code", "research", "safety"})
MAX_MESSAGES = 64
MAX_PROMPT_CHARS = 12000
_ROLES = frozenset({"system", "user", "assistant"})


class ContractError(ValueError):
    """Invalid client request; carries an OpenAI-style error code."""

    def __init__(self, message: str, code: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.code = code


def error_body(message: str, code: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": code}}


def list_models(available: Mapping[str, str], statuses: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Model list with honest availability; never reports an untrained model as ready."""
    statuses = statuses or {}
    data = []
    for model_id, capability in MODEL_IDS.items():
        ready = capability in available
        data.append({
            "id": model_id,
            "object": "model",
            "owned_by": "aethrion",
            "capability": capability,
            "available": ready,
            "status": "ready" if ready else statuses.get(capability, "not_trained"),
        })
    return {"object": "list", "data": data}


def parse_chat_request(payload: Any) -> tuple[str, str, dict[str, Any]]:
    """Validate a chat-completions payload -> (capability, prompt, options)."""
    if not isinstance(payload, Mapping):
        raise ContractError("Request body must be a JSON object.")
    model = payload.get("model")
    if not isinstance(model, str) or model not in MODEL_IDS:
        raise ContractError(f"Unknown model. Use one of: {', '.join(MODEL_IDS)}.", "model_not_found")
    capability = MODEL_IDS[model]
    if capability not in CHAT_COMPLETION_CAPABILITIES:
        raise ContractError(
            f"Model '{model}' does not support chat completions; use its dedicated endpoint.",
            "unsupported_model_endpoint",
        )
    if payload.get("stream"):
        raise ContractError("Streaming is not implemented yet.", "stream_not_supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise ContractError(f"'messages' must be a list of 1-{MAX_MESSAGES} items.")
    lines: list[str] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise ContractError("Each message must be an object.")
        role, content = item.get("role"), item.get("content")
        if role not in _ROLES or not isinstance(content, str) or not content.strip():
            raise ContractError("Each message needs a valid role and non-empty string content.")
        lines.append(f"{role}: {content.strip()}")
    prompt = "\n".join(lines)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ContractError(f"Prompt exceeds {MAX_PROMPT_CHARS} characters.", "context_length_exceeded")
    options: dict[str, Any] = {}
    max_tokens = payload.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
            raise ContractError("'max_tokens' must be an integer between 1 and 2048.")
        options["max_new_tokens"] = max_tokens
    temperature = payload.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ContractError("'temperature' must be a number between 0 and 2.")
        options["temperature"] = float(temperature)
    return capability, prompt, options


def output_text(output: Any) -> str:
    """Extract assistant text from a handler result without guessing structure."""
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("text", "answer", "content", "output"):
            if isinstance(output.get(key), str):
                return output[key]
    raise ContractError("Model returned an unsupported output shape.", "invalid_model_output")


def build_chat_response(model: str, output: Any, prompt: str) -> dict[str, Any]:
    text = output_text(output)
    if not text.strip():
        raise ContractError("Model returned empty output.", "invalid_model_output")
    # Token counts are character-based estimates; they are labeled as such.
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"estimated": True, "prompt_chars": len(prompt), "completion_chars": len(text)},
    }
