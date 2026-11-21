# -*- coding: utf-8 -*-
"""Original, owner-authored code samples for the CODE model (written for AETHRION-LLM)."""
import json
S = [
("python-temel","def toplam(a, b):\n    \"\"\"İki sayının toplamını döndürür.\"\"\"\n    return a + b\n\n\nif __name__ == '__main__':\n    print(toplam(2, 3))\n"),
("python-temel","def ortalama(sayilar):\n    \"\"\"Boş olmayan bir sayı listesinin ortalamasını hesaplar.\"\"\"\n    if not sayilar:\n        raise ValueError('liste boş olamaz')\n    return sum(sayilar) / len(sayilar)\n"),
("python-temel","def faktoriyel(n):\n    if n < 0:\n        raise ValueError('n negatif olamaz')\n    sonuc = 1\n    for i in range(2, n + 1):\n        sonuc *= i\n    return sonuc\n"),
("python-temel","def palindrom_mu(metin):\n    temiz = ''.join(c.lower() for c in metin if c.isalnum())\n    return temiz == temiz[::-1]\n"),
("python-temel","def fibonacci(n):\n    a, b = 0, 1\n    dizi = []\n    for _ in range(n):\n        dizi.append(a)\n        a, b = b, a + b\n    return dizi\n"),
("python-temel","def kelime_say(metin):\n    sayac = {}\n    for kelime in metin.lower().split():\n        sayac[kelime] = sayac.get(kelime, 0) + 1\n    return sayac\n"),
("python-temel","def en_buyuk(sayilar):\n    if not sayilar:\n        return None\n    en = sayilar[0]\n    for s in sayilar[1:]:\n        if s > en:\n            en = s\n    return en\n"),
("python-temel","def asal_mi(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n"),
("python-algoritma","def ikili_arama(dizi, hedef):\n    sol, sag = 0, len(dizi) - 1\n    while sol <= sag:\n        orta = (sol + sag) // 2\n        if dizi[orta] == hedef:\n            return orta\n        if dizi[orta] < hedef:\n            sol = orta + 1\n        else:\n            sag = orta - 1\n    return -1\n"),
("python-algoritma","def kabarcik_sirala(dizi):\n    dizi = list(dizi)\n    n = len(dizi)\n    for i in range(n):\n        degisti = False\n        for j in range(n - 1 - i):\n            if dizi[j] > dizi[j + 1]:\n                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]\n                degisti = True\n        if not degisti:\n            break\n    return dizi\n"),
("python-algoritma","def birlestir_sirala(dizi):\n    if len(dizi) <= 1:\n        return list(dizi)\n    orta = len(dizi) // 2\n    sol = birlestir_sirala(dizi[:orta])\n    sag = birlestir_sirala(dizi[orta:])\n    sonuc, i, j = [], 0, 0\n    while i < len(sol) and j < len(sag):\n        if sol[i] <= sag[j]:\n            sonuc.append(sol[i]); i += 1\n        else:\n            sonuc.append(sag[j]); j += 1\n    return sonuc + sol[i:] + sag[j:]\n"),
("python-algoritma","def ebob(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n"),
("python-sinif","class Hesap:\n    def __init__(self, bakiye=0):\n        self._bakiye = bakiye\n\n    def yatir(self, miktar):\n        if miktar <= 0:\n            raise ValueError('miktar pozitif olmalı')\n        self._bakiye += miktar\n\n    def cek(self, miktar):\n        if miktar > self._bakiye:\n            raise ValueError('yetersiz bakiye')\n        self._bakiye -= miktar\n\n    @property\n    def bakiye(self):\n        return self._bakiye\n"),
("python-sinif","class Yigin:\n    def __init__(self):\n        self._ogeler = []\n\n    def ekle(self, oge):\n        self._ogeler.append(oge)\n\n    def cikar(self):\n        if not self._ogeler:\n            raise IndexError('yığın boş')\n        return self._ogeler.pop()\n\n    def bos_mu(self):\n        return not self._ogeler\n"),
("python-dosya","import json\n\n\ndef json_oku(yol):\n    with open(yol, encoding='utf-8') as f:\n        return json.load(f)\n\n\ndef json_yaz(yol, veri):\n    with open(yol, 'w', encoding='utf-8') as f:\n        json.dump(veri, f, ensure_ascii=False, indent=2)\n"),
("python-dosya","def satirlari_say(yol):\n    sayi = 0\n    with open(yol, encoding='utf-8') as f:\n        for _ in f:\n            sayi += 1\n    return sayi\n"),
("python-test","import unittest\n\n\ndef topla(a, b):\n    return a + b\n\n\nclass ToplaTesti(unittest.TestCase):\n    def test_pozitif(self):\n        self.assertEqual(topla(2, 3), 5)\n\n    def test_negatif(self):\n        self.assertEqual(topla(-1, -1), -2)\n\n\nif __name__ == '__main__':\n    unittest.main()\n"),
("javascript-temel","function topla(a, b) {\n  return a + b;\n}\n\nconsole.log(topla(2, 3));\n"),
("javascript-temel","const ortalama = (sayilar) => {\n  if (sayilar.length === 0) throw new Error('liste boş');\n  return sayilar.reduce((t, s) => t + s, 0) / sayilar.length;\n};\n"),
("javascript-temel","function tekrarsiz(dizi) {\n  return [...new Set(dizi)];\n}\n"),
("javascript-async","async function veriGetir(adres) {\n  const yanit = await fetch(adres);\n  if (!yanit.ok) {\n    throw new Error('istek başarısız: ' + yanit.status);\n  }\n  return yanit.json();\n}\n"),
("javascript-dom","document.querySelector('#gonder').addEventListener('click', () => {\n  const ad = document.querySelector('#ad').value.trim();\n  if (!ad) return;\n  document.querySelector('#mesaj').textContent = 'Merhaba ' + ad;\n});\n"),
("sql","SELECT musteri_id, COUNT(*) AS siparis_sayisi\nFROM siparisler\nWHERE durum = 'tamamlandi'\nGROUP BY musteri_id\nHAVING COUNT(*) > 3\nORDER BY siparis_sayisi DESC;\n"),
("sql","CREATE TABLE kullanicilar (\n  id INTEGER PRIMARY KEY,\n  ad TEXT NOT NULL,\n  eposta TEXT NOT NULL UNIQUE,\n  olusturma_tarihi TEXT NOT NULL\n);\n"),
("bash","#!/usr/bin/env bash\nset -euo pipefail\n\nfor dosya in *.txt; do\n  echo \"$dosya: $(wc -l < \"$dosya\") satır\"\ndone\n"),
("html","<!doctype html>\n<html lang=\"tr\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Merhaba</title>\n</head>\n<body>\n  <h1>Merhaba Dünya</h1>\n  <p>İlk sayfam.</p>\n</body>\n</html>\n"),
("css",".kart {\n  display: flex;\n  flex-direction: column;\n  gap: 12px;\n  padding: 16px;\n  border-radius: 12px;\n  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);\n}\n"),
]
with open('data/authored/code/records.jsonl','w',encoding='utf-8') as f:
    for i,(g,t) in enumerate(S,1):
        f.write(json.dumps({"id":f"code-{i:04d}","group_id":f"code-{g}-{(i-1)//2}","text":t},ensure_ascii=False)+"\n")
print(len(S))
