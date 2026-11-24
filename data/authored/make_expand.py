# -*- coding: utf-8 -*-
"""Deterministic expansion of first-party (owner-attested) data: research pairs, chat Q&A, extra BASE topics.
Run from repo root: python3 data/authored/make_expand.py   (idempotent; appends only missing ids)"""
import importlib.util, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_base", ROOT / "base" / "make_base.py")
mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)

ONES = ["sıfır","bir","iki","üç","dört","beş","altı","yedi","sekiz","dokuz"]
TENS = ["","on","yirmi","otuz","kırk","elli","altmış","yetmiş","seksen","doksan"]
def words(n):
    if n < 10: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    h, r = divmod(n, 100)
    head = "yüz" if h == 1 else ONES[h] + " yüz"
    return head + ("" if r == 0 else " " + words(r))

CITIES = [("İstanbul","Marmara",34),("Ankara","İç Anadolu",6),("İzmir","Ege",35),("Bursa","Marmara",16),("Antalya","Akdeniz",7),
 ("Adana","Akdeniz",1),("Konya","İç Anadolu",42),("Trabzon","Karadeniz",61),("Erzurum","Doğu Anadolu",25),("Gaziantep","Güneydoğu Anadolu",27),
 ("Samsun","Karadeniz",55),("Eskişehir","İç Anadolu",26),("Mersin","Akdeniz",33),("Diyarbakır","Güneydoğu Anadolu",21),("Kayseri","İç Anadolu",38),
 ("Van","Doğu Anadolu",65),("Edirne","Marmara",22),("Çanakkale","Marmara",17),("Muğla","Ege",48),("Denizli","Ege",20)]
ANIMALS = [("Kedi","memeli","kedigiller"),("Köpek","memeli","evcil"),("At","memeli","otçul"),("İnek","memeli","süt veren"),("Koyun","memeli","yün veren"),
 ("Yunus","memeli","denizde yaşayan"),("Balina","memeli","denizde yaşayan"),("Yarasa","memeli","uçabilen"),("Kartal","kuş","yırtıcı"),("Serçe","kuş","küçük"),
 ("Penguen","kuş","uçamayan"),("Baykuş","kuş","gece avlanan"),("Tavuk","kuş","evcil"),("Yılan","sürüngen","bacaksız"),("Kaplumbağa","sürüngen","kabuklu"),
 ("Timsah","sürüngen","su kenarında yaşayan"),("Kertenkele","sürüngen","küçük"),("Somon","balık","göç eden"),("Köpekbalığı","balık","yırtıcı"),("Hamsi","balık","sürü halinde yaşayan"),
 ("Kurbağa","iki yaşamlı","suda ve karada yaşayan"),("Semender","iki yaşamlı","nemli yerleri seven"),("Arı","böcek","bal yapan"),("Karınca","böcek","koloni halinde yaşayan"),("Kelebek","böcek","kanatlı")]
MONTHS = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"]
DAYS = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"]

def extra_groups():
    g = {}
    g["cografya"] = []
    for c, r, n in CITIES:
        g["cografya"] += [f"{c}, {r} bölgesinde yer alan bir ilimizdir.", f"{c} ilinin plaka kodu {words(n)} sayısıdır."]
    g["hayvanlar"] = []
    for a, k, f in ANIMALS:
        g["hayvanlar"] += [f"{a}, {k} sınıfında yer alan bir canlıdır.", f"{a} için en bilinen özellik, {f} olmasıdır."]
    cal = []
    for i, m in enumerate(MONTHS, 1):
        cal.append(f"{m}, yılın {words(i)}. ayıdır.")
    for i, d in enumerate(DAYS, 1):
        cal.append(f"{d}, haftanın {words(i)}. günüdür.")
    g["takvim"] = cal
    mat = []
    for a in range(2, 30, 3):
        for b in range(3, 22, 4):
            mat.append(f"{words(a)} artı {words(b)} {words(a+b)} eder.")
            mat.append(f"{words(a)} çarpı {words(b)} {words(a*b)} eder.")
            if a > b: mat.append(f"{words(a)} eksi {words(b)} {words(a-b)} eder.")
    g["matematik"] = mat
    return g

def subject(sentence):
    head = sentence.split(",")[0].strip() if "," in sentence else ""
    head = re.sub(r"^Bir ", "", head)
    return head if 0 < len(head.split()) <= 4 else None

def all_facts():
    facts = []
    for topic, text in mb.GROUPS.items():
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for i, l in enumerate(lines): facts.append((topic, i, l, lines))
    for topic, lines in extra_groups().items():
        for i, l in enumerate(lines): facts.append((topic, i, l, lines))
    return facts

def append_jsonl(path, rows):
    have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    new = [r for r in rows if r["id"] not in have]
    with path.open("a", encoding="utf-8") as f:
        for r in new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)

def main():
    research, chat = [], []
    for topic, i, sent, lines in all_facts():
        s = subject(sent)
        if not s: continue
        ctx = " ".join(lines[max(0, i - 1): i + 2])
        gid = f"fact-{topic}-{i:03d}"
        qs = [f"{s} nedir?", f"{s} hakkında kısaca bilgi verir misin?", f"{s} ile ilgili temel bilgi nedir?"]
        for k, q in enumerate(qs):
            research.append({"id": f"research-fact-{topic}-{i:03d}-{k}", "group_id": gid, "query": q, "passage": ctx})
        for k, q in enumerate(qs[:2]):
            chat.append({"id": f"chat-fact-{topic}-{i:03d}-{k}", "group_id": gid, "prompt": q, "response": sent})
    print("research +", append_jsonl(ROOT / "research" / "records.jsonl", research))
    print("chat +", append_jsonl(ROOT / "chat" / "records.jsonl", chat))
    # arithmetic Q&A for chat
    arith = []
    for a in range(2, 30, 3):
        for b in range(3, 22, 4):
            arith.append({"id": f"chat-arith-{a}-{b}", "group_id": f"fact-aritmetik-{a}-{b}",
                          "prompt": f"{words(a)} artı {words(b)} kaç eder?", "response": f"{words(a)} artı {words(b)} {words(a+b)} eder."})
    print("chat arith +", append_jsonl(ROOT / "chat" / "records.jsonl", arith))

if __name__ == "__main__":
    main()
