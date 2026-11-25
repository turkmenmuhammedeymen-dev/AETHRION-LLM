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
) -> str:
    available = set(available)
    plan = decide(message)
    sources: list[dict] = []
    for url in _URL.findall(message)[:2]:
        try:
            sources.append(fetch_page(url))
        except WebFetchError as exc:
            sources.append({"url": url, "title": "", "text": "", "error": str(exc)})
    prompt = message
    ok = [s for s in sources if s.get("text")]
    if ok:
        prompt += "\n\nKaynaklar:\n" + "\n".join(f"[{s['url']}] {s['text'][:1500]}" for s in ok)

    if plan.capability != "chat":
        if plan.capability not in available:
            return (f"Bu istek AETHRION-{plan.capability.upper()} modeline yönlendirildi, ancak bu model henüz "
                    "eğitilip yayınlanmadı. Başka bir model yerine geçirilmedi.")
        result = str(dispatch(plan.capability, prompt))
        if "chat" in available:
            return chat("Uzman model sonucunu kullanıcıya açıkla:\n" + result)
        return result
    if "chat" not in available:
        if ok:
            return "AETHRION sohbet modeli henüz yayınlanmadı. Okunan kaynak:\n" + "\n".join(f"- {s['title'] or s['url']}: {s['url']}" for s in ok)
        return "AETHRION's own trained checkpoint/tokenizer is not loaded. No third-party model is silently substituted."
    answer = chat(prompt)
    if ok:
        answer += "\n\nKaynaklar: " + ", ".join(s["url"] for s in ok)
    return answer
