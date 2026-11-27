"""Intent detection so the chat model can hand a request to the right specialist.

Pure Python, deterministic and auditable. Delegation only *selects* a capability;
dispatch still goes through CapabilityRouter / the native registry, which refuse
models that are not trained and released (no fake fallback).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code", ("kod yaz", "fonksiyon yaz", "python", "javascript", "hata ayıkla", "debug", "algoritma", "sql ", "script")),
    ("image", ("resim çiz", "görsel üret", "görsel oluştur", "fotoğraf üret", "illüstrasyon", "logo tasarla")),
    ("3d", ("3d model", "3 boyutlu", "üç boyutlu", "nokta bulutu")),
    ("tts", ("sesli oku", "seslendir", "metni konuştur", "text to speech")),
    ("asr", ("sesi yazıya", "ses kaydını", "transkript", "konuşmayı yazıya")),
    ("vision", ("bu fotoğrafta", "bu resimde", "görseli analiz", "resmi açıkla", "görselde ne")),
    ("document", ("belgeyi", "pdf", "dosyayı özetle", "tablodan", "sözleşme", "fatura")),
    ("research", ("araştır", "kaynak bul", "literatür", "güncel haber", "internette ara", "web'de ara", "son durum")),
)
_URL = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class Delegation:
    capability: str
    reason: str
    needs_web: bool = False


def decide(prompt: str) -> Delegation:
    text = " ".join(prompt.lower().split())
    needs_web = bool(_URL.search(text)) or any(k in text for k in ("internette ara", "web'de ara", "güncel", "son dakika", "bugün"))
    for capability, keys in _RULES:
        hit = next((k for k in keys if k in text), None)
        if hit:
            return Delegation(capability, f"keyword:{hit}", needs_web)
    return Delegation("chat", "default", needs_web)
