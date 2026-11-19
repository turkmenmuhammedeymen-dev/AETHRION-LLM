"""Main-chat intent orchestration for the eleven-model AETHRION family.

CHAT is the default conversational model. High-confidence task cues delegate
work to a released specialist. SAFE remains a mandatory policy gate in the
capability handlers; VOCODER is an internal dependency of TTS, not a user-facing
intent. Missing modality inputs fail clearly instead of silently falling back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from aethrion.routing import Capability, CapabilityRequest, CapabilityRouter


class AssistantRoutingError(RuntimeError):
    """Base error for assistant orchestration."""


class AssistantInputRequired(AssistantRoutingError):
    """The selected specialist requires an attachment or structured input."""


@dataclass(frozen=True)
class Intent:
    capability: Capability
    reason: str
    confidence: float


@dataclass(frozen=True)
class AssistantResponse:
    capability: Capability
    delegated: bool
    reason: str
    provider: str
    output: Any


# Deterministic high-precision cues. Uncertain requests remain with CHAT.
_RULES = (
    (Capability.CODE, (
        "write code", "write a function", "code this", "debug", "fix this code",
        "implement", "python code", "javascript code", "refactor", "kod yaz",
        "kodla", "kodu düzelt", "hata ayıkla", "fonksiyon yaz", "programla",
    )),
    (Capability.RESEARCH, (
        "research", "find sources", "compare sources", "literature review",
        "araştır", "kaynak bul", "kaynakları karşılaştır", "literatür taraması",
        "akademik kaynak", "kanıtları incele",
    )),
    (Capability.IMAGE, (
        "generate an image", "create an image", "draw an image", "make an image",
        "görsel oluştur", "resim oluştur", "çizim oluştur", "bir görsel çiz",
    )),
    (Capability.THREE_D, (
        "3d model", "3d asset", "create a mesh", "point cloud",
        "3 boyutlu model", "3d nesne", "üç boyutlu model", "mesh oluştur",
    )),
    (Capability.ASR, (
        "transcribe audio", "transcribe this", "speech to text", "voice to text",
        "sesi yazıya çevir", "ses kaydını yaz", "konuşmayı metne çevir",
    )),
    (Capability.TTS, (
        "read this aloud", "speak this", "generate speech", "text to speech",
        "seslendir", "sesli oku", "konuşma üret", "metni sese çevir",
    )),
    (Capability.DOCUMENT, (
        "analyze this document", "summarize this pdf", "extract from this file",
        "document analysis", "belgeyi analiz et", "pdf'i özetle", "dosyayı incele",
        "belgeden çıkar", "dokümanı analiz et",
    )),
    (Capability.VISION, (
        "describe this image", "what is in this image", "read this image",
        "image question", "görüntüyü açıkla", "resimde ne var", "fotoğrafı incele",
        "görseli analiz et",
    )),
)


def classify_intent(prompt: str, options: Mapping[str, Any] | None = None) -> Intent:
    """Select a specialist only for clear task cues; otherwise use main CHAT."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty.")
    opts = options or {}
    requested = opts.get("capability")
    if requested is not None:
        try:
            capability = Capability(str(requested).lower())
        except ValueError as exc:
            raise ValueError(f"Unknown requested capability: {requested}") from exc
        if capability in {Capability.SAFETY}:
            raise ValueError("SAFE is a policy gate, not a user-selectable task route.")
        return Intent(capability, "explicit_capability_selection", 1.0)

    normalized = " ".join(prompt.casefold().split())
    matches = []
    for capability, phrases in _RULES:
        if any(phrase in normalized for phrase in phrases):
            matches.append(capability)
    if len(matches) == 1:
        return Intent(matches[0], "high_confidence_task_cue", 0.9)
    # Ambiguous multi-task prompts stay in CHAT; no arbitrary specialist choice.
    if len(matches) > 1:
        return Intent(Capability.CHAT, "ambiguous_multiple_task_cues", 0.5)
    return Intent(Capability.CHAT, "general_conversation_or_uncertain_intent", 0.8)


_REQUIRED_INPUTS = {
    Capability.VISION: "image_path",
    Capability.ASR: "audio_path",
    Capability.DOCUMENT: "document_path",
}


class AssistantOrchestrator:
    """Route one user turn to CHAT or a specialist through the verified router."""

    def __init__(self, router: CapabilityRouter):
        self.router = router

    def dispatch(
        self,
        prompt: str,
        options: Mapping[str, Any] | None = None,
    ) -> AssistantResponse:
        opts = dict(options or {})
        intent = classify_intent(prompt, opts)
        missing = _REQUIRED_INPUTS.get(intent.capability)
        if missing and not opts.get(missing):
            raise AssistantInputRequired(
                f"AETHRION-{intent.capability.value.upper()} requires '{missing}'. "
                "Attach the required input or select another task."
            )
        # Routing metadata is not forwarded to the specialist model.
        opts.pop("capability", None)
        result = self.router.dispatch(
            intent.capability,
            CapabilityRequest(prompt, opts),
        )
        return AssistantResponse(
            capability=intent.capability,
            delegated=intent.capability != Capability.CHAT,
            reason=intent.reason,
            provider=result.provider,
            output=result.output,
        )
