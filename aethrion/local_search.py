"""First-party keyword search (BM25) over pages the user or the app has fetched.

No third-party search API is called. Documents enter the index only through
``add`` (e.g. after ``web_fetch.fetch`` passed its public-URL checks), so every
hit carries the URL it came from. It cannot discover pages on its own.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[0-9a-zçğıöşü]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower().replace("İ", "i").replace("I", "ı")) if len(t) > 1]


class LocalSearch:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: dict[str, tuple[str, Counter, int]] = {}
        self._text: dict[str, str] = {}

    def add(self, url: str, title: str, text: str) -> None:
        toks = _tokens(f"{title} {text}")
        self._docs[url] = (title, Counter(toks), len(toks))
        self._text[url] = text

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        q = set(_tokens(query))
        n = len(self._docs)
        if not q or not n:
            return []
        avg = sum(d[2] for d in self._docs.values()) / n or 1.0
        out = []
        for url, (title, tf, length) in self._docs.items():
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                df = sum(1 for d in self._docs.values() if t in d[1])
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / avg))
            if score > 0:
                out.append({"url": url, "title": title, "text": self._text[url][:1500], "score": score})
        return sorted(out, key=lambda r: -r["score"])[:limit]
