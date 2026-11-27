from aethrion.chat_orchestrator import orchestrate
from aethrion.local_search import LocalSearch


def _index():
    s = LocalSearch()
    s.add("https://a.example/ay", "Ay", "Ay Dünya'nın doğal uydusudur ve gelgitleri etkiler.")
    s.add("https://a.example/gunes", "Güneş", "Güneş bir yıldızdır ve sistemin merkezidir.")
    return s


def test_ranks_relevant_document_first():
    hits = _index().search("gelgitleri ne etkiler")
    assert hits and hits[0]["url"] == "https://a.example/ay"


def test_empty_index_or_no_match_returns_nothing():
    assert LocalSearch().search("ay") == []
    assert _index().search("zzzz") == []


def test_orchestrator_cites_local_hit():
    out = orchestrate("güncel: gelgitleri ne etkiler", {"chat"}, lambda c, p: "", lambda p: "cevap", search=_index())
    assert "https://a.example/ay" in out
