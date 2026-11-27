"""Chat entry point that delegates to specialists and reads live pages.

Selection only: every specialist call still goes through the router/registry
gates, and an unavailable model is reported honestly (no substitute model).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from aethrion.delegation import decide
from aethrion.web_fetch import WebFetchError, fetch

_URL = re.compile(r"https?://[^\s)>\]]+")


def orchestrate(
    message: str,
    available: Iterable[str],
    dispatch: Callable[[str, str], str],
    chat: Callable[[str], str],
    fetch_page: Callable[[str], dict] = fetch,
    trace: list[str] | None = None,
) -> str:
    """Route one message. If ``trace`` is a list, it receives the real steps taken
    (thinking mode): routing decision, pages read, model availability. It records
    actions only, never invented reasoning."""
    def note(text: str) -> None:
        if trace is not None:
            trace.append(text)

    available = set(available)
    plan = decide(message)
    note(f"Yönlendirme: {plan.capability} ({plan.reason})")
    sources: list[dict] = []
    for url in _URL.findall(message)[:2]:
        try:
            sources.append(fetch_page(url))
            note(f"Sayfa okundu: {url}")
        except WebFetchError as exc:
            note(f"Sayfa okunamadı: {url} ({exc})")
            sources.append({"url": url, "title": "", "text": "", "error": str(exc)})
    prompt = message
    ok = [s for s in sources if s.get("text")]
    if ok:
        prompt += "\n\nKaynaklar:\n" + "\n".join(f"[{s['url']}] {s['text'][:1500]}" for s in ok)

    if plan.capability != "chat":
        if plan.capability not in available:
            note(f"{plan.capability} modeli yayında değil; başka model kullanılmadı")
            return (f"Bu istek AETHRION-{plan.capability.upper()} modeline yönlendirildi, ancak bu model henüz "
                    "eğitilip yayınlanmadı. Başka bir model yerine geçirilmedi.")
        note(f"{plan.capability} uzman modeline gönderildi")
        result = str(dispatch(plan.capability, prompt))
        if "chat" in available:
            return chat("Uzman model sonucunu kullanıcıya açıkla:\n" + result)
        return result
    if "chat" not in available:
        note("chat modeli yayında değil")
        if ok:
            return "AETHRION sohbet modeli henüz yayınlanmadı. Okunan kaynak:\n" + "\n".join(f"- {s['title'] or s['url']}: {s['url']}" for s in ok)
        return "AETHRION's own trained checkpoint/tokenizer is not loaded. No third-party model is silently substituted."
    note("Yanıt AETHRION sohbet modeliyle üretildi")
    answer = chat(prompt)
    if ok:
        answer += "\n\nKaynaklar: " + ", ".join(s["url"] for s in ok)
    return answer
