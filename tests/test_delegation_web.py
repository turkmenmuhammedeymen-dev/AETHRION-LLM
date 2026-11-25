import pytest
from aethrion.delegation import decide
from aethrion.web_fetch import WebFetchError, check_url, extract_text


def test_delegation_routes_by_intent():
    assert decide("Bana python ile bir fonksiyon yaz").capability == "code"
    assert decide("Bu pdf dosyasını özetle").capability == "document"
    assert decide("Merhaba nasılsın").capability == "chat"
    assert decide("internette ara: deprem son durum").needs_web


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.com", "http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "https://user:p@example.com/"])
def test_web_fetch_blocks_unsafe(url):
    with pytest.raises(WebFetchError):
        check_url(url)


def test_extract_text_drops_scripts():
    title, text = extract_text("<html><title>T</title><script>x()</script><p>Merhaba</p></html>")
    assert title == "T" and text == "Merhaba"
