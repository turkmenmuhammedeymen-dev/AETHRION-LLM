from aethrion.chat_orchestrator import orchestrate


def test_trace_records_real_steps_and_unavailable_model():
    trace = []
    out = orchestrate("kod yaz: iki sayıyı topla", set(), lambda c, p: "x", lambda p: "y", trace=trace)
    assert trace and trace[0].startswith("Yönlendirme:")
    assert "yayında değil" in " ".join(trace) or "yayınlanmadı" in out


def test_trace_optional_and_chat_path():
    trace = []
    assert orchestrate("merhaba", {"chat"}, lambda c, p: "x", lambda p: "selam", trace=trace) == "selam"
    assert trace[-1].startswith("Yanıt")
    assert orchestrate("merhaba", {"chat"}, lambda c, p: "x", lambda p: "selam") == "selam"
