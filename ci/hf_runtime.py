"""Read-only Hugging Face Space runtime and Gradio health verification."""
from __future__ import annotations

import json
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SpaceRuntimeVerificationError(RuntimeError):
    """Raised when the deployed Space does not become healthy."""


def wait_for_space_running(
    api: Any,
    repo_id: str,
    *,
    timeout_seconds: int = 480,
    poll_interval: int = 10,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Wait for a Space build to finish without changing hardware or runtime state."""
    if timeout_seconds < 1 or poll_interval < 1:
        raise ValueError("timeout_seconds and poll_interval must be positive")
    deadline = monotonic() + timeout_seconds
    while True:
        runtime = api.get_space_runtime(repo_id)
        raw_stage = getattr(runtime, "stage", "UNKNOWN")
        stage = str(getattr(raw_stage, "value", raw_stage)).upper()
        print(f"HF_SPACE_STAGE={stage}", flush=True)
        if stage == "RUNNING":
            return stage
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            raise SpaceRuntimeVerificationError(
                f"Hugging Face Space entered {stage}; inspect its build/runtime logs."
            )
        if stage in {"PAUSED", "SLEEPING"}:
            raise SpaceRuntimeVerificationError(
                f"Hugging Face Space is {stage}; refusing to change its runtime state automatically."
            )
        if monotonic() >= deadline:
            raise SpaceRuntimeVerificationError(
                f"Timed out waiting for Hugging Face Space; last stage was {stage}."
            )
        sleep(poll_interval)


def probe_gradio_config(
    repo_id: str,
    *,
    timeout_seconds: int = 20,
    open_url: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Check the public Gradio config endpoint and validate its response shape."""
    if "/" not in repo_id or not all(repo_id.split("/", 1)):
        raise ValueError("repo_id must use owner/name format")
    owner, name = repo_id.split("/", 1)
    host = f"{owner}-{name}".lower()
    url = f"https://{host}.hf.space/config"
    opener = open_url or urlopen
    request = Request(url, headers={"User-Agent": "AETHRION-Space-Healthcheck/1.0"})
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise SpaceRuntimeVerificationError(
                    f"Gradio config endpoint returned HTTP {status}."
                )
            payload = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpaceRuntimeVerificationError(
            f"Gradio config endpoint health check failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SpaceRuntimeVerificationError("Gradio config response is not a JSON object.")
    if not isinstance(payload.get("components"), list):
        raise SpaceRuntimeVerificationError("Gradio config is missing its components list.")
    if not isinstance(payload.get("dependencies"), list):
        raise SpaceRuntimeVerificationError("Gradio config is missing its dependencies list.")
    return payload
