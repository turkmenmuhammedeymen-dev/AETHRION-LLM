# -*- coding: utf-8 -*-
"""Round 2: more original owner-authored code samples. Appends to records.jsonl (idempotent by id)."""
import json
P = "data/authored/code/records.jsonl"
M = [
("python-veri","def gruplandir(ogeler, anahtar):\n    gruplar = {}\n    for oge in ogeler:\n        gruplar.setdefault(anahtar(oge), []).append(oge)\n    return gruplar\n"),
("python-veri","def duzlestir(ic_ice):\n    return [oge for alt in ic_ice for oge in alt]\n"),
("python-veri","def tekrar_edenler(dizi):\n    gorulen, tekrar = set(), set()\n    for oge in dizi:\n        if oge in gorulen:\n            tekrar.add(oge)\n        gorulen.add(oge)\n    return sorted(tekrar)\n"),
("python-veri","def parcalara_bol(dizi, boyut):\n    if boyut <= 0:\n        raise ValueError('boyut pozitif olmalı')\n    return [dizi[i:i + boyut] for i in range(0, len(dizi), boyut)]\n"),
("python-veri","def sozluk_birlestir(a, b):\n    sonuc = dict(a)\n    for k, v in b.items():\n        if k in sonuc and isinstance(sonuc[k], dict) and isinstance(v, dict):\n            sonuc[k] = sozluk_birlestir(sonuc[k], v)\n        else:\n            sonuc[k] = v\n    return sonuc\n"),
("python-metin","def baslik_yap(metin):\n    return ' '.join(k[:1].upper() + k[1:].lower() for k in metin.split())\n"),
("python-metin","def anagram_mi(a, b):\n    temiz = lambda s: sorted(c.lower() for c in s if c.isalpha())\n    return temiz(a) == temiz(b)\n"),
("python-metin","def en_uzun_kelime(metin):\n    kelimeler = metin.split()\n    return max(kelimeler, key=len) if kelimeler else ''\n"),
("python-metin","import re\n\n\ndef sayilari_bul(metin):\n    return [int(s) for s in re.findall(r'-?\\d+', metin)]\n"),
("python-metin","def sezar_sifrele(metin, kaydirma):\n    sonuc = []\n    for c in metin:\n        if 'a' <= c <= 'z':\n            sonuc.append(chr((ord(c) - 97 + kaydirma) % 26 + 97))\n        elif 'A' <= c <= 'Z':\n            sonuc.append(chr((ord(c) - 65 + kaydirma) % 26 + 65))\n        else:\n            sonuc.append(c)\n    return ''.join(sonuc)\n"),
("python-algoritma","def hizli_sirala(dizi):\n    if len(dizi) <= 1:\n        return list(dizi)\n    pivot = dizi[len(dizi) // 2]\n    kucuk = [x for x in dizi if x < pivot]\n    esit = [x for x in dizi if x == pivot]\n    buyuk = [x for x in dizi if x > pivot]\n    return hizli_sirala(kucuk) + esit + hizli_sirala(buyuk)\n"),
("python-algoritma","def eratosthenes(n):\n    if n < 2:\n        return []\n    asal = [True] * (n + 1)\n    asal[0] = asal[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if asal[i]:\n            for j in range(i * i, n + 1, i):\n                asal[j] = False\n    return [i for i, a in enumerate(asal) if a]\n"),
("python-algoritma","from collections import deque\n\n\ndef genislik_oncelikli(graf, baslangic):\n    gorulen, sira, sonuc = {baslangic}, deque([baslangic]), []\n    while sira:\n        dugum = sira.popleft()\n        sonuc.append(dugum)\n        for komsu in graf.get(dugum, []):\n            if komsu not in gorulen:\n                gorulen.add(komsu)\n                sira.append(komsu)\n    return sonuc\n"),
("python-algoritma","def en_uzun_ortak_alt_dizi(a, b):\n    tablo = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n    for i in range(1, len(a) + 1):\n        for j in range(1, len(b) + 1):\n            if a[i - 1] == b[j - 1]:\n                tablo[i][j] = tablo[i - 1][j - 1] + 1\n            else:\n                tablo[i][j] = max(tablo[i - 1][j], tablo[i][j - 1])\n    return tablo[-1][-1]\n"),
("python-algoritma","def ikinci_derece_kok(a, b, c):\n    delta = b * b - 4 * a * c\n    if delta < 0:\n        return []\n    k = delta ** 0.5\n    return sorted({(-b - k) / (2 * a), (-b + k) / (2 * a)})\n"),
("python-sinif","class Kuyruk:\n    def __init__(self):\n        self._ogeler = []\n\n    def ekle(self, oge):\n        self._ogeler.append(oge)\n\n    def al(self):\n        if not self._ogeler:\n            raise IndexError('kuyruk boş')\n        return self._ogeler.pop(0)\n\n    def __len__(self):\n        return len(self._ogeler)\n"),
("python-sinif","class Nokta:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\n    def uzaklik(self, diger):\n        return ((self.x - diger.x) ** 2 + (self.y - diger.y) ** 2) ** 0.5\n\n    def __repr__(self):\n        return f'Nokta({self.x}, {self.y})'\n"),
("python-sinif","from dataclasses import dataclass\n\n\n@dataclass\nclass Urun:\n    ad: str\n    fiyat: float\n    adet: int = 1\n\n    def toplam(self):\n        return self.fiyat * self.adet\n"),
("python-sinif","class Sayac:\n    def __init__(self):\n        self.degerler = {}\n\n    def ekle(self, anahtar):\n        self.degerler[anahtar] = self.degerler.get(anahtar, 0) + 1\n\n    def en_cok(self, n=3):\n        return sorted(self.degerler.items(), key=lambda kv: -kv[1])[:n]\n"),
("python-hata","def guvenli_bol(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None\n"),
("python-hata","def tam_sayiya_cevir(metin, varsayilan=0):\n    try:\n        return int(metin)\n    except (TypeError, ValueError):\n        return varsayilan\n"),
("python-hata","import time\n\n\ndef tekrar_dene(islev, deneme=3, bekleme=0.5):\n    for i in range(deneme):\n        try:\n            return islev()\n        except Exception:\n            if i == deneme - 1:\n                raise\n            time.sleep(bekleme)\n"),
("python-dosya","import csv\n\n\ndef csv_oku(yol):\n    with open(yol, newline='', encoding='utf-8') as f:\n        return list(csv.DictReader(f))\n"),
("python-dosya","from pathlib import Path\n\n\ndef uzantiya_gore_say(klasor):\n    sayac = {}\n    for yol in Path(klasor).rglob('*'):\n        if yol.is_file():\n            sayac[yol.suffix] = sayac.get(yol.suffix, 0) + 1\n    return sayac\n"),
("python-uretec","def kareler(n):\n    for i in range(n):\n        yield i * i\n"),
("python-uretec","def satir_oku(yol):\n    with open(yol, encoding='utf-8') as f:\n        for satir in f:\n            yield satir.rstrip('\\n')\n"),
("python-test","import unittest\n\n\ndef palindrom_mu(m):\n    t = ''.join(c.lower() for c in m if c.isalnum())\n    return t == t[::-1]\n\n\nclass PalindromTesti(unittest.TestCase):\n    def test_dogru(self):\n        self.assertTrue(palindrom_mu('Kayak'))\n\n    def test_yanlis(self):\n        self.assertFalse(palindrom_mu('Merhaba'))\n\n\nif __name__ == '__main__':\n    unittest.main()\n"),
("python-test","import unittest\n\n\ndef ortalama(s):\n    if not s:\n        raise ValueError('boş')\n    return sum(s) / len(s)\n\n\nclass OrtalamaTesti(unittest.TestCase):\n    def test_normal(self):\n        self.assertEqual(ortalama([2, 4]), 3)\n\n    def test_bos(self):\n        with self.assertRaises(ValueError):\n            ortalama([])\n\n\nif __name__ == '__main__':\n    unittest.main()\n"),
("javascript-temel","const gruplandir = (liste, anahtar) =>\n  liste.reduce((acc, oge) => {\n    (acc[oge[anahtar]] ||= []).push(oge);\n    return acc;\n  }, {});\n"),
("javascript-temel","function tersle(metin) {\n  return [...metin].reverse().join('');\n}\n"),
("javascript-temel","const sirala = (dizi) => [...dizi].sort((a, b) => a - b);\n"),
("javascript-temel","function debounce(islev, bekleme) {\n  let zamanlayici;\n  return (...args) => {\n    clearTimeout(zamanlayici);\n    zamanlayici = setTimeout(() => islev(...args), bekleme);\n  };\n}\n"),
("javascript-sinif","class Yigin {\n  #ogeler = [];\n  ekle(oge) { this.#ogeler.push(oge); }\n  cikar() {\n    if (this.#ogeler.length === 0) throw new Error('yığın boş');\n    return this.#ogeler.pop();\n  }\n  get boyut() { return this.#ogeler.length; }\n}\n"),
("javascript-async","const bekle = (ms) => new Promise((coz) => setTimeout(coz, ms));\n\nasync function tekrarDene(islev, deneme = 3) {\n  for (let i = 0; i < deneme; i++) {\n    try {\n      return await islev();\n    } catch (hata) {\n      if (i === deneme - 1) throw hata;\n      await bekle(300);\n    }\n  }\n}\n"),
("javascript-dom","const liste = document.querySelector('#liste');\nfor (const ad of ['Ali', 'Ayşe', 'Mehmet']) {\n  const oge = document.createElement('li');\n  oge.textContent = ad;\n  liste.appendChild(oge);\n}\n"),
("sql","SELECT u.ad, SUM(s.tutar) AS toplam\nFROM urunler u\nJOIN siparis_kalemleri s ON s.urun_id = u.id\nGROUP BY u.ad\nORDER BY toplam DESC\nLIMIT 10;\n"),
("sql","UPDATE kullanicilar\nSET aktif = 0\nWHERE son_giris < DATE('now', '-365 days');\n"),
("sql","CREATE INDEX idx_siparis_musteri ON siparisler (musteri_id, olusturma_tarihi);\n"),
("sql","SELECT d.ad, AVG(c.maas) AS ortalama_maas\nFROM departmanlar d\nLEFT JOIN calisanlar c ON c.departman_id = d.id\nGROUP BY d.ad;\n"),
("bash","#!/usr/bin/env bash\nset -euo pipefail\n\nyedek=\"yedek-$(date +%Y%m%d)\"\nmkdir -p \"$yedek\"\ncp -r ./belgeler \"$yedek/\"\necho \"yedek hazır: $yedek\"\n"),
("bash","#!/usr/bin/env bash\nset -euo pipefail\n\nif [ $# -lt 1 ]; then\n  echo \"kullanım: $0 <klasör>\" >&2\n  exit 1\nfi\ndu -sh \"$1\"/* | sort -h\n"),
("html","<form action=\"/gonder\" method=\"post\">\n  <label for=\"eposta\">E-posta</label>\n  <input id=\"eposta\" name=\"eposta\" type=\"email\" required>\n  <button type=\"submit\">Gönder</button>\n</form>\n"),
("html","<ul class=\"menu\">\n  <li><a href=\"/\">Ana sayfa</a></li>\n  <li><a href=\"/hakkimizda\">Hakkımızda</a></li>\n  <li><a href=\"/iletisim\">İletişim</a></li>\n</ul>\n"),
("css","@media (max-width: 600px) {\n  .menu {\n    flex-direction: column;\n    gap: 8px;\n  }\n}\n"),
("css",":root {\n  --ana-renk: #1f3a5f;\n  --vurgu: #c9a227;\n}\n\n.dugme {\n  background: var(--ana-renk);\n  color: white;\n  border: 0;\n  padding: 10px 18px;\n  border-radius: 8px;\n}\n.dugme:hover { background: var(--vurgu); }\n"),
("json-yaml","{\n  \"ad\": \"ornek-proje\",\n  \"surum\": \"1.0.0\",\n  \"betikler\": {\n    \"test\": \"python -m unittest\"\n  }\n}\n"),
("json-yaml","servis:\n  ad: web\n  port: 8080\n  ortam:\n    - DEBUG=0\n    - LOG_SEVIYESI=info\n"),
]
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
have = {r["id"] for r in rows}
start = len(rows)
for k, (g, t) in enumerate(M, 1):
    i = start + k
    rid = f"code-{i:04d}"
    if rid in have:
        continue
    rows.append({"id": rid, "group_id": f"code-{g}-r2-{(k-1)//2}", "text": t})
with open(P, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows))
