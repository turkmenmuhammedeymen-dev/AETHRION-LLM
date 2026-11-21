# Review sheet: data/authored/code/records.jsonl

27 records. Mark problems by id; nothing here approves training.

### code-0001 (code-python-temel-0)
- **text**: def toplam(a, b):
    """İki sayının toplamını döndürür."""
    return a + b


if __name__ == '__main__':
    print(toplam(2, 3))


### code-0002 (code-python-temel-0)
- **text**: def ortalama(sayilar):
    """Boş olmayan bir sayı listesinin ortalamasını hesaplar."""
    if not sayilar:
        raise ValueError('liste boş olamaz')
    return sum(sayilar) / len(sayilar)


### code-0003 (code-python-temel-1)
- **text**: def faktoriyel(n):
    if n < 0:
        raise ValueError('n negatif olamaz')
    sonuc = 1
    for i in range(2, n + 1):
        sonuc *= i
    return sonuc


### code-0004 (code-python-temel-1)
- **text**: def palindrom_mu(metin):
    temiz = ''.join(c.lower() for c in metin if c.isalnum())
    return temiz == temiz[::-1]


### code-0005 (code-python-temel-2)
- **text**: def fibonacci(n):
    a, b = 0, 1
    dizi = []
    for _ in range(n):
        dizi.append(a)
        a, b = b, a + b
    return dizi


### code-0006 (code-python-temel-2)
- **text**: def kelime_say(metin):
    sayac = {}
    for kelime in metin.lower().split():
        sayac[kelime] = sayac.get(kelime, 0) + 1
    return sayac


### code-0007 (code-python-temel-3)
- **text**: def en_buyuk(sayilar):
    if not sayilar:
        return None
    en = sayilar[0]
    for s in sayilar[1:]:
        if s > en:
            en = s
    return en


### code-0008 (code-python-temel-3)
- **text**: def asal_mi(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


### code-0009 (code-python-algoritma-4)
- **text**: def ikili_arama(dizi, hedef):
    sol, sag = 0, len(dizi) - 1
    while sol <= sag:
        orta = (sol + sag) // 2
        if dizi[orta] == hedef:
            return orta
        if dizi[orta] < hedef:
            sol = orta + 1
        else:
            sag = orta - 1
    return -1


### code-0010 (code-python-algoritma-4)
- **text**: def kabarcik_sirala(dizi):
    dizi = list(dizi)
    n = len(dizi)
    for i in range(n):
        degisti = False
        for j in range(n - 1 - i):
            if dizi[j] > dizi[j + 1]:
                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]
                degisti = True
        if not degisti:
            break
    return dizi


### code-0011 (code-python-algoritma-5)
- **text**: def birlestir_sirala(dizi):
    if len(dizi) <= 1:
        return list(dizi)
    orta = len(dizi) // 2
    sol = birlestir_sirala(dizi[:orta])
    sag = birlestir_sirala(dizi[orta:])
    sonuc, i, j = [], 0, 0
    while i < len(sol) and j < len(sag):
        if sol[i] <= sag[j]:
            sonuc.append(sol[i]); i += 1
        else:
            sonuc.append(sag[j]); j += 1
    return sonuc + sol[i:] + sag[j:]


### code-0012 (code-python-algoritma-5)
- **text**: def ebob(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


### code-0013 (code-python-sinif-6)
- **text**: class Hesap:
    def __init__(self, bakiye=0):
        self._bakiye = bakiye

    def yatir(self, miktar):
        if miktar <= 0:
            raise ValueError('miktar pozitif olmalı')
        self._bakiye += miktar

    def cek(self, miktar):
        if miktar > self._bakiye:
            raise ValueError('yetersiz bakiye')
        self._bakiye -= miktar

    @property
    def bakiye(self):
        return self._bakiye


### code-0014 (code-python-sinif-6)
- **text**: class Yigin:
    def __init__(self):
        self._ogeler = []

    def ekle(self, oge):
        self._ogeler.append(oge)

    def cikar(self):
        if not self._ogeler:
            raise IndexError('yığın boş')
        return self._ogeler.pop()

    def bos_mu(self):
        return not self._ogeler


### code-0015 (code-python-dosya-7)
- **text**: import json


def json_oku(yol):
    with open(yol, encoding='utf-8') as f:
        return json.load(f)


def json_yaz(yol, veri):
    with open(yol, 'w', encoding='utf-8') as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)


### code-0016 (code-python-dosya-7)
- **text**: def satirlari_say(yol):
    sayi = 0
    with open(yol, encoding='utf-8') as f:
        for _ in f:
            sayi += 1
    return sayi


### code-0017 (code-python-test-8)
- **text**: import unittest


def topla(a, b):
    return a + b


class ToplaTesti(unittest.TestCase):
    def test_pozitif(self):
        self.assertEqual(topla(2, 3), 5)

    def test_negatif(self):
        self.assertEqual(topla(-1, -1), -2)


if __name__ == '__main__':
    unittest.main()


### code-0018 (code-javascript-temel-8)
- **text**: function topla(a, b) {
  return a + b;
}

console.log(topla(2, 3));


### code-0019 (code-javascript-temel-9)
- **text**: const ortalama = (sayilar) => {
  if (sayilar.length === 0) throw new Error('liste boş');
  return sayilar.reduce((t, s) => t + s, 0) / sayilar.length;
};


### code-0020 (code-javascript-temel-9)
- **text**: function tekrarsiz(dizi) {
  return [...new Set(dizi)];
}


### code-0021 (code-javascript-async-10)
- **text**: async function veriGetir(adres) {
  const yanit = await fetch(adres);
  if (!yanit.ok) {
    throw new Error('istek başarısız: ' + yanit.status);
  }
  return yanit.json();
}


### code-0022 (code-javascript-dom-10)
- **text**: document.querySelector('#gonder').addEventListener('click', () => {
  const ad = document.querySelector('#ad').value.trim();
  if (!ad) return;
  document.querySelector('#mesaj').textContent = 'Merhaba ' + ad;
});


### code-0023 (code-sql-11)
- **text**: SELECT musteri_id, COUNT(*) AS siparis_sayisi
FROM siparisler
WHERE durum = 'tamamlandi'
GROUP BY musteri_id
HAVING COUNT(*) > 3
ORDER BY siparis_sayisi DESC;


### code-0024 (code-sql-11)
- **text**: CREATE TABLE kullanicilar (
  id INTEGER PRIMARY KEY,
  ad TEXT NOT NULL,
  eposta TEXT NOT NULL UNIQUE,
  olusturma_tarihi TEXT NOT NULL
);


### code-0025 (code-bash-12)
- **text**: #!/usr/bin/env bash
set -euo pipefail

for dosya in *.txt; do
  echo "$dosya: $(wc -l < "$dosya") satır"
done


### code-0026 (code-html-12)
- **text**: <!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Merhaba</title>
</head>
<body>
  <h1>Merhaba Dünya</h1>
  <p>İlk sayfam.</p>
</body>
</html>


### code-0027 (code-css-13)
- **text**: .kart {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

