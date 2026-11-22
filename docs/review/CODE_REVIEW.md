# Review sheet: data/authored/code/records.jsonl

74 records. Mark problems by id; nothing here approves training.

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


### code-0028 (code-python-veri-r2-0)
- **text**: def gruplandir(ogeler, anahtar):
    gruplar = {}
    for oge in ogeler:
        gruplar.setdefault(anahtar(oge), []).append(oge)
    return gruplar


### code-0029 (code-python-veri-r2-0)
- **text**: def duzlestir(ic_ice):
    return [oge for alt in ic_ice for oge in alt]


### code-0030 (code-python-veri-r2-1)
- **text**: def tekrar_edenler(dizi):
    gorulen, tekrar = set(), set()
    for oge in dizi:
        if oge in gorulen:
            tekrar.add(oge)
        gorulen.add(oge)
    return sorted(tekrar)


### code-0031 (code-python-veri-r2-1)
- **text**: def parcalara_bol(dizi, boyut):
    if boyut <= 0:
        raise ValueError('boyut pozitif olmalı')
    return [dizi[i:i + boyut] for i in range(0, len(dizi), boyut)]


### code-0032 (code-python-veri-r2-2)
- **text**: def sozluk_birlestir(a, b):
    sonuc = dict(a)
    for k, v in b.items():
        if k in sonuc and isinstance(sonuc[k], dict) and isinstance(v, dict):
            sonuc[k] = sozluk_birlestir(sonuc[k], v)
        else:
            sonuc[k] = v
    return sonuc


### code-0033 (code-python-metin-r2-2)
- **text**: def baslik_yap(metin):
    return ' '.join(k[:1].upper() + k[1:].lower() for k in metin.split())


### code-0034 (code-python-metin-r2-3)
- **text**: def anagram_mi(a, b):
    temiz = lambda s: sorted(c.lower() for c in s if c.isalpha())
    return temiz(a) == temiz(b)


### code-0035 (code-python-metin-r2-3)
- **text**: def en_uzun_kelime(metin):
    kelimeler = metin.split()
    return max(kelimeler, key=len) if kelimeler else ''


### code-0036 (code-python-metin-r2-4)
- **text**: import re


def sayilari_bul(metin):
    return [int(s) for s in re.findall(r'-?\d+', metin)]


### code-0037 (code-python-metin-r2-4)
- **text**: def sezar_sifrele(metin, kaydirma):
    sonuc = []
    for c in metin:
        if 'a' <= c <= 'z':
            sonuc.append(chr((ord(c) - 97 + kaydirma) % 26 + 97))
        elif 'A' <= c <= 'Z':
            sonuc.append(chr((ord(c) - 65 + kaydirma) % 26 + 65))
        else:
            sonuc.append(c)
    return ''.join(sonuc)


### code-0038 (code-python-algoritma-r2-5)
- **text**: def hizli_sirala(dizi):
    if len(dizi) <= 1:
        return list(dizi)
    pivot = dizi[len(dizi) // 2]
    kucuk = [x for x in dizi if x < pivot]
    esit = [x for x in dizi if x == pivot]
    buyuk = [x for x in dizi if x > pivot]
    return hizli_sirala(kucuk) + esit + hizli_sirala(buyuk)


### code-0039 (code-python-algoritma-r2-5)
- **text**: def eratosthenes(n):
    if n < 2:
        return []
    asal = [True] * (n + 1)
    asal[0] = asal[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if asal[i]:
            for j in range(i * i, n + 1, i):
                asal[j] = False
    return [i for i, a in enumerate(asal) if a]


### code-0040 (code-python-algoritma-r2-6)
- **text**: from collections import deque


def genislik_oncelikli(graf, baslangic):
    gorulen, sira, sonuc = {baslangic}, deque([baslangic]), []
    while sira:
        dugum = sira.popleft()
        sonuc.append(dugum)
        for komsu in graf.get(dugum, []):
            if komsu not in gorulen:
                gorulen.add(komsu)
                sira.append(komsu)
    return sonuc


### code-0041 (code-python-algoritma-r2-6)
- **text**: def en_uzun_ortak_alt_dizi(a, b):
    tablo = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                tablo[i][j] = tablo[i - 1][j - 1] + 1
            else:
                tablo[i][j] = max(tablo[i - 1][j], tablo[i][j - 1])
    return tablo[-1][-1]


### code-0042 (code-python-algoritma-r2-7)
- **text**: def ikinci_derece_kok(a, b, c):
    delta = b * b - 4 * a * c
    if delta < 0:
        return []
    k = delta ** 0.5
    return sorted({(-b - k) / (2 * a), (-b + k) / (2 * a)})


### code-0043 (code-python-sinif-r2-7)
- **text**: class Kuyruk:
    def __init__(self):
        self._ogeler = []

    def ekle(self, oge):
        self._ogeler.append(oge)

    def al(self):
        if not self._ogeler:
            raise IndexError('kuyruk boş')
        return self._ogeler.pop(0)

    def __len__(self):
        return len(self._ogeler)


### code-0044 (code-python-sinif-r2-8)
- **text**: class Nokta:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def uzaklik(self, diger):
        return ((self.x - diger.x) ** 2 + (self.y - diger.y) ** 2) ** 0.5

    def __repr__(self):
        return f'Nokta({self.x}, {self.y})'


### code-0045 (code-python-sinif-r2-8)
- **text**: from dataclasses import dataclass


@dataclass
class Urun:
    ad: str
    fiyat: float
    adet: int = 1

    def toplam(self):
        return self.fiyat * self.adet


### code-0046 (code-python-sinif-r2-9)
- **text**: class Sayac:
    def __init__(self):
        self.degerler = {}

    def ekle(self, anahtar):
        self.degerler[anahtar] = self.degerler.get(anahtar, 0) + 1

    def en_cok(self, n=3):
        return sorted(self.degerler.items(), key=lambda kv: -kv[1])[:n]


### code-0047 (code-python-hata-r2-9)
- **text**: def guvenli_bol(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None


### code-0048 (code-python-hata-r2-10)
- **text**: def tam_sayiya_cevir(metin, varsayilan=0):
    try:
        return int(metin)
    except (TypeError, ValueError):
        return varsayilan


### code-0049 (code-python-hata-r2-10)
- **text**: import time


def tekrar_dene(islev, deneme=3, bekleme=0.5):
    for i in range(deneme):
        try:
            return islev()
        except Exception:
            if i == deneme - 1:
                raise
            time.sleep(bekleme)


### code-0050 (code-python-dosya-r2-11)
- **text**: import csv


def csv_oku(yol):
    with open(yol, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


### code-0051 (code-python-dosya-r2-11)
- **text**: from pathlib import Path


def uzantiya_gore_say(klasor):
    sayac = {}
    for yol in Path(klasor).rglob('*'):
        if yol.is_file():
            sayac[yol.suffix] = sayac.get(yol.suffix, 0) + 1
    return sayac


### code-0052 (code-python-uretec-r2-12)
- **text**: def kareler(n):
    for i in range(n):
        yield i * i


### code-0053 (code-python-uretec-r2-12)
- **text**: def satir_oku(yol):
    with open(yol, encoding='utf-8') as f:
        for satir in f:
            yield satir.rstrip('\n')


### code-0054 (code-python-test-r2-13)
- **text**: import unittest


def palindrom_mu(m):
    t = ''.join(c.lower() for c in m if c.isalnum())
    return t == t[::-1]


class PalindromTesti(unittest.TestCase):
    def test_dogru(self):
        self.assertTrue(palindrom_mu('Kayak'))

    def test_yanlis(self):
        self.assertFalse(palindrom_mu('Merhaba'))


if __name__ == '__main__':
    unittest.main()


### code-0055 (code-python-test-r2-13)
- **text**: import unittest


def ortalama(s):
    if not s:
        raise ValueError('boş')
    return sum(s) / len(s)


class OrtalamaTesti(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(ortalama([2, 4]), 3)

    def test_bos(self):
        with self.assertRaises(ValueError):
            ortalama([])


if __name__ == '__main__':
    unittest.main()


### code-0056 (code-javascript-temel-r2-14)
- **text**: const gruplandir = (liste, anahtar) =>
  liste.reduce((acc, oge) => {
    (acc[oge[anahtar]] ||= []).push(oge);
    return acc;
  }, {});


### code-0057 (code-javascript-temel-r2-14)
- **text**: function tersle(metin) {
  return [...metin].reverse().join('');
}


### code-0058 (code-javascript-temel-r2-15)
- **text**: const sirala = (dizi) => [...dizi].sort((a, b) => a - b);


### code-0059 (code-javascript-temel-r2-15)
- **text**: function debounce(islev, bekleme) {
  let zamanlayici;
  return (...args) => {
    clearTimeout(zamanlayici);
    zamanlayici = setTimeout(() => islev(...args), bekleme);
  };
}


### code-0060 (code-javascript-sinif-r2-16)
- **text**: class Yigin {
  #ogeler = [];
  ekle(oge) { this.#ogeler.push(oge); }
  cikar() {
    if (this.#ogeler.length === 0) throw new Error('yığın boş');
    return this.#ogeler.pop();
  }
  get boyut() { return this.#ogeler.length; }
}


### code-0061 (code-javascript-async-r2-16)
- **text**: const bekle = (ms) => new Promise((coz) => setTimeout(coz, ms));

async function tekrarDene(islev, deneme = 3) {
  for (let i = 0; i < deneme; i++) {
    try {
      return await islev();
    } catch (hata) {
      if (i === deneme - 1) throw hata;
      await bekle(300);
    }
  }
}


### code-0062 (code-javascript-dom-r2-17)
- **text**: const liste = document.querySelector('#liste');
for (const ad of ['Ali', 'Ayşe', 'Mehmet']) {
  const oge = document.createElement('li');
  oge.textContent = ad;
  liste.appendChild(oge);
}


### code-0063 (code-sql-r2-17)
- **text**: SELECT u.ad, SUM(s.tutar) AS toplam
FROM urunler u
JOIN siparis_kalemleri s ON s.urun_id = u.id
GROUP BY u.ad
ORDER BY toplam DESC
LIMIT 10;


### code-0064 (code-sql-r2-18)
- **text**: UPDATE kullanicilar
SET aktif = 0
WHERE son_giris < DATE('now', '-365 days');


### code-0065 (code-sql-r2-18)
- **text**: CREATE INDEX idx_siparis_musteri ON siparisler (musteri_id, olusturma_tarihi);


### code-0066 (code-sql-r2-19)
- **text**: SELECT d.ad, AVG(c.maas) AS ortalama_maas
FROM departmanlar d
LEFT JOIN calisanlar c ON c.departman_id = d.id
GROUP BY d.ad;


### code-0067 (code-bash-r2-19)
- **text**: #!/usr/bin/env bash
set -euo pipefail

yedek="yedek-$(date +%Y%m%d)"
mkdir -p "$yedek"
cp -r ./belgeler "$yedek/"
echo "yedek hazır: $yedek"


### code-0068 (code-bash-r2-20)
- **text**: #!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "kullanım: $0 <klasör>" >&2
  exit 1
fi
du -sh "$1"/* | sort -h


### code-0069 (code-html-r2-20)
- **text**: <form action="/gonder" method="post">
  <label for="eposta">E-posta</label>
  <input id="eposta" name="eposta" type="email" required>
  <button type="submit">Gönder</button>
</form>


### code-0070 (code-html-r2-21)
- **text**: <ul class="menu">
  <li><a href="/">Ana sayfa</a></li>
  <li><a href="/hakkimizda">Hakkımızda</a></li>
  <li><a href="/iletisim">İletişim</a></li>
</ul>


### code-0071 (code-css-r2-21)
- **text**: @media (max-width: 600px) {
  .menu {
    flex-direction: column;
    gap: 8px;
  }
}


### code-0072 (code-css-r2-22)
- **text**: :root {
  --ana-renk: #1f3a5f;
  --vurgu: #c9a227;
}

.dugme {
  background: var(--ana-renk);
  color: white;
  border: 0;
  padding: 10px 18px;
  border-radius: 8px;
}
.dugme:hover { background: var(--vurgu); }


### code-0073 (code-json-yaml-r2-22)
- **text**: {
  "ad": "ornek-proje",
  "surum": "1.0.0",
  "betikler": {
    "test": "python -m unittest"
  }
}


### code-0074 (code-json-yaml-r2-23)
- **text**: servis:
  ad: web
  port: 8080
  ortam:
    - DEBUG=0
    - LOG_SEVIYESI=info

