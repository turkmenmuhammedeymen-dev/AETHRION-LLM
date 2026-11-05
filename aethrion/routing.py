"""Explicit capability routing for AETHRION's specialist model backends.

The base text model cannot generate images or 3D assets by itself. A capability
is available only after a real handler has been registered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class Capability(str, Enum):
    CHAT = "chat"
    CODE = "code"
    IMAGE = "image"
    THREE_D = "3d"
    RESEARCH = "research"
    FILE_ANALYSIS = "file_analysis"
    AUDIO = "audio"


@dataclass(frozen=True)
class CapabilityRequest:
    prompt: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityResponse:
    capability: Capability
    provider: str
    output: Any


CapabilityHandler = Callable[[CapabilityRequest], Any]


class CapabilityError(RuntimeError):
    """Base error for capability selection and dispatch."""


class CapabilityUnavailable(CapabilityError):
    """Raised when no real provider is registered for a selected capability."""


class CapabilityRouter:
    """Dispatch only to an explicitly registered handler; never fake a fallback."""

    def __init__(self) -> None:
        self._handlers: dict[Capability, tuple[str, CapabilityHandler]] = {}

    def register(
        self,
        capability: Capability | str,
        provider: str,
        handler: CapabilityHandler,
        *,
        replace: bool = False,
    ) -> None:
        key = Capability(capability)
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty identifier")
        if not callable(handler):
            raise TypeError("handler must be callable")
        if key in self._handlers and not replace:
            raise ValueError(f"A handler is already registered for {key.value}")
        self._handlers[key] = (provider.strip(), handler)

    def available(self) -> dict[str, str]:
        """Return capability-to-provider metadata, never credentials."""
        return {key.value: value[0] for key, value in self._handlers.items()}

    def dispatch(
        self,
        capability: Capability | str,
        request: CapabilityRequest,
    ) -> CapabilityResponse:
        key = Capability(capability)
        if not isinstance(request, CapabilityRequest):
            raise TypeError("request must be a CapabilityRequest")
        if not request.prompt.strip():
            raise ValueError("prompt must not be empty")
        registered = self._handlers.get(key)
        if registered is None:
            raise CapabilityUnavailable(
                f"No provider is configured for capability '{key.value}'."
            )
        provider, handler = registered
        result = handler(request)
        if result is None:
            raise CapabilityError(
                f"Provider '{provider}' returned no result for '{key.value}'."
            )
        return CapabilityResponse(capability=key, provider=provider, output=result)
