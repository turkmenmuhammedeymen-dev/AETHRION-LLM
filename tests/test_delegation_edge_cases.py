from aethrion.delegation import decide


def test_script_inside_transcript_is_not_code():
    assert decide("please transcript this audio").capability != "code"
    assert decide("bir python script yaz").capability == "code"


def test_bugun_alone_is_not_web_but_fact_questions_are():
    assert decide("bugün nasılsın").needs_web is False
    assert decide("bugün hava nasıl").needs_web is True
    assert decide("bugünkü dolar kuru ne").needs_web is True


def test_specialists_still_route():
    assert decide("seslendir: merhaba").capability == "tts"
    assert decide("bu fotoğrafta ne var").capability == "vision"
    assert decide("resim çiz kedi").capability == "image"
