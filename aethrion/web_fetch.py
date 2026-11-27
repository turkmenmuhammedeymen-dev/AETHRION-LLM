"""First-party live web access (no third-party search API).

Fetches a public http(s) page directly and returns plain text with its URL, so
answers can cite where information came from. Private/loopback/link-local
addresses are refused (SSRF protection); size and time are capped.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

MAX_BYTES = 1_000_000
TIMEOUT = 10


class WebFetchError(RuntimeError):
    pass


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip += 1
        self._in_title = tag == "title"

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
        elif not self.skip and data.strip():
            self.parts.append(data.strip())


def check_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise WebFetchError("Only plain public http(s) URLs are allowed.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise WebFetchError(f"Host could not be resolved: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebFetchError("Refusing to fetch a non-public address.")
    return url


def extract_text(html: str) -> tuple[str, str]:
    parser = _Text()
    parser.feed(html)
    return parser.title, " ".join(parser.parts)


def fetch(url: str, max_chars: int = 6000) -> dict[str, str]:
    check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "AETHRION-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - scheme validated above
            if "html" not in resp.headers.get("Content-Type", "") and "text" not in resp.headers.get("Content-Type", ""):
                raise WebFetchError("Only text/html pages are supported.")
            raw = resp.read(MAX_BYTES)
            final = resp.geturl()
    except OSError as exc:
        raise WebFetchError(f"Fetch failed: {exc}") from exc
    check_url(final)
    title, text = extract_text(raw.decode("utf-8", "replace"))
    return {"url": final, "title": title, "text": text[:max_chars]}
