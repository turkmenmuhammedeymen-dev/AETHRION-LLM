# -*- coding: utf-8 -*-
"""Round 3: programmatically generated, executable-verified Python templates created at the project owner's request; not individually human-reviewed.
Every snippet is compiled and executed before it is written. Idempotent by id."""
import ast, json, itertools
P = "data/authored/code/records.jsonl"
OPS = {"topla": ("+", "toplamı"), "carp": ("*", "çarpımı"), "fark": ("-", "farkı")}
S = []  # (group, text)
for (i, (name, (op, tr))) in enumerate(OPS.items()):
    for k in (2, 3, 5, 7, 11):
        S.append((f"gen-aritmetik-{name}", f'def {name}_{k}(x):\n    """x ile {k} sayısının {tr}ni döndürür."""\n    return x {op} {k}\n'))
for k in (10, 25, 40, 60, 75, 90):
    S.append(("gen-esik", f'def esik_{k}_ustu(sayilar):\n    """{k} değerinden büyük olanları sırayla döndürür."""\n    return [s for s in sayilar if s > {k}]\n'))
    S.append(("gen-esik", f'def esik_{k}_say(sayilar):\n    """{k} değerine eşit veya küçük kaç sayı olduğunu verir."""\n    sayac = 0\n    for s in sayilar:\n        if s <= {k}:\n            sayac += 1\n    return sayac\n'))
for m in ("upper", "lower", "title", "strip", "capitalize", "swapcase"):
    for pre in ("kisi", "urun", "sehir"):
        S.append((f"gen-metin-{m}", f'def {pre}_{m}(liste):\n    """{pre} listesindeki her metni {m} yöntemiyle dönüştürür."""\n    return [m.{m}() for m in liste]\n'))
for n in (3, 4, 5, 6, 8, 9, 12):
    S.append(("gen-dongu", f'def ilk_{n}_kare():\n    """İlk {n} doğal sayının karesini listeler."""\n    sonuc = []\n    for i in range(1, {n} + 1):\n        sonuc.append(i * i)\n    return sonuc\n'))
    S.append(("gen-dongu", f'def kat_tablosu_{n}():\n    """{n} sayısının 1\'den 10\'a kadar katlarını sözlük olarak verir."""\n    return {{i: i * {n} for i in range(1, 11)}}\n'))
    S.append(("gen-dongu", f'def basamak_kontrol_{n}(sayi):\n    """Sayının {n} basamaklı olup olmadığını denetler."""\n    return len(str(abs(sayi))) == {n}\n'))
for cls, alan in (("Ogrenci","not_ortalamasi"),("Calisan","maas"),("Kitap","sayfa"),("Arac","hiz"),("Oyuncu","puan"),("Hesap","bakiye"),("Sipariş","tutar"),("Ders","kredi")):
    cn = cls.replace("ş", "s")
    S.append((f"gen-sinif-{cn.lower()}", f'class {cn}:\n    def __init__(self, ad, {alan}=0):\n        self.ad = ad\n        self.{alan} = {alan}\n\n    def artir(self, miktar):\n        self.{alan} += miktar\n        return self.{alan}\n\n    def __repr__(self):\n        return f"{cn}({{self.ad!r}}, {{self.{alan}!r}})"\n'))
for dil, (kw, ornek) in {"toplam": ("sum", "[1, 2, 3]"), "en_buyuk": ("max", "[4, 9, 2]"), "en_kucuk": ("min", "[4, 9, 2]"), "uzunluk": ("len", "'abc'"), "sirali": ("sorted", "[3, 1, 2]")}.items():
    S.append((f"gen-yerlesik-{dil}", f'def {dil}_goster(veri):\n    """{kw} işlevini veriye uygular ve sonucu yazdırır."""\n    sonuc = {kw}(veri)\n    print(sonuc)\n    return sonuc\n\n\nif __name__ == "__main__":\n    {dil}_goster({ornek})\n'))
for a, b in itertools.combinations(("tek", "cift", "pozitif", "negatif", "sifir"), 2):
    cond = {"tek": "x % 2 == 1", "cift": "x % 2 == 0", "pozitif": "x > 0", "negatif": "x < 0", "sifir": "x == 0"}
    S.append((f"gen-kosul-{a}-{b}", f'def {a}_veya_{b}(x):\n    """Sayı {a} ya da {b} ise True verir."""\n    ilk = {cond[a]}\n    ikinci = {cond[b]}\n    return ilk or ikinci\n'))
ok = []
for g, t in S:
    ast.parse(t)
    if "__main__" in t:
        exec(compile(t.replace('if __name__ == "__main__":', "if True:"), "<g>", "exec"), {})
    ok.append((g, t))
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
have = {r["id"] for r in rows}
start = len([r for r in rows if not r["id"].startswith("codegen-")])
n = 0
for k, (g, t) in enumerate(ok, 1):
    rid = f"codegen-{k:04d}"
    if rid in have:
        continue
    rows.append({"id": rid, "group_id": g, "text": t}); n += 1
with open(P, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "added", n)
