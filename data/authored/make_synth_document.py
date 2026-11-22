# -*- coding: utf-8 -*-
"""Procedurally generated DOCUMENT data: fictional forms with token-aligned boxes/layout/tags (first-party; no real people).
Run from repo root: python3 data/authored/make_synth_document.py [seed]"""
import json, sys, random
from pathlib import Path
from aethrion.specialist_prepare import _split
from aethrion.tokenizer import ByteBPETokenizer
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
VOCAB, MAXLEN = 2048, 512
# layout ids: 1 başlık, 2 etiket, 3 değer, 4 tablo başlığı, 5 tablo hücresi, 6 altbilgi
# tag ids: 0 diğer, 1 tarih, 2 tutar, 3 ad, 4 numara, 5 şehir, 6 ürün, 7 adet
AD = ["Ayşe Demo", "Ali Örnek", "Zeynep Deneme", "Mehmet Kurgu", "Elif Model", "Can Taslak", "Deniz Test", "Selin Örnek"]
SEHIR = ["Ankara", "İzmir", "Bursa", "Konya", "Adana", "Samsun", "Trabzon", "Eskişehir"]
URUN = ["kalem", "defter", "silgi", "dosya", "makas", "bant", "zımba", "cetvel", "kitap", "çanta"]
FIRMA = ["Örnek Kırtasiye", "Deneme Ticaret", "Kurgu Market", "Model Gıda", "Taslak Teknik"]

def doc_fatura(r):
    L = [(1, [(f"{r.choice(FIRMA)} FATURA", 0)]),
         (2, [("Fatura", 0), ("No:", 0), (f"{r.randint(1000,9999)}", 4)])]
    L.append((2, [("Tarih:", 0), (f"{r.randint(1,28):02d}.{r.randint(1,12):02d}.2026", 1)]))
    L.append((2, [("Müşteri:", 0)] + [(w, 3) for w in r.choice(AD).split()]))
    L.append((2, [("Şehir:", 0), (r.choice(SEHIR), 5)]))
    L.append((4, [("Ürün", 0), ("Adet", 0), ("Tutar", 0)]))
    top = 0
    for _ in range(r.randint(2, 4)):
        a, t = r.randint(1, 9), r.randint(5, 90); top += a * t
        L.append((5, [(r.choice(URUN), 6), (str(a), 7), (f"{a*t} TL", 2)]))
    L.append((2, [("Toplam:", 0), (f"{top} TL", 2)]))
    L.append((6, [("Teşekkür", 0), ("ederiz.", 0)]))
    return L
def doc_basvuru(r):
    L = [(1, [("BAŞVURU", 0), ("FORMU", 0)]),
         (2, [("Ad", 0), ("Soyad:", 0)] + [(w, 3) for w in r.choice(AD).split()]),
         (2, [("Başvuru", 0), ("No:", 0), (f"{r.randint(100,999)}", 4)]),
         (2, [("Tarih:", 0), (f"{r.randint(1,28):02d}.{r.randint(1,12):02d}.2026", 1)]),
         (2, [("Şehir:", 0), (r.choice(SEHIR), 5)]),
         (2, [("Talep", 0), ("edilen", 0), ("ürün:", 0), (r.choice(URUN), 6)]),
         (2, [("Adet:", 0), (str(r.randint(1, 20)), 7)]),
         (6, [("Bu", 0), ("form", 0), ("kurgusaldır.", 0)])]
    return L
def doc_irsaliye(r):
    L = [(1, [("SEVK", 0), ("İRSALİYESİ", 0)]), (2, [("Sevk", 0), ("No:", 0), (f"{r.randint(10000,99999)}", 4)]),
         (2, [("Tarih:", 0), (f"{r.randint(1,28):02d}.{r.randint(1,12):02d}.2026", 1)]),
         (2, [("Teslim", 0), ("şehri:", 0), (r.choice(SEHIR), 5)]), (4, [("Ürün", 0), ("Adet", 0)])]
    for _ in range(r.randint(2, 5)):
        L.append((5, [(r.choice(URUN), 6), (str(r.randint(1, 50)), 7)]))
    L.append((6, [("Teslim", 0), ("alan:", 0)] + [(w, 3) for w in r.choice(AD).split()]))
    return L
KINDS = [doc_fatura, doc_basvuru, doc_irsaliye]

def render(L):
    """Return text and per-byte (box, layout, tag) attributes."""
    n = len(L); lh = 1.0 / (n + 1); text = ""; spans = []
    width = max(sum(len(w) + 1 for w, _ in ws) for _, ws in L) or 1
    for li, (lay, ws) in enumerate(L):
        x = 0
        for w, tag in ws:
            start = len(text.encode("utf-8")); text += w; end = len(text.encode("utf-8"))
            box = [round(x / (width + 1), 4), round((li + .5) * lh, 4), round(min(1.0, (x + len(w)) / (width + 1)), 4), round((li + 1) * lh, 4)]
            spans.append((start, end, box, lay, tag)); text += " "; x += len(w) + 1
        text = text.rstrip(" ") + "\n"
    return text.rstrip("\n"), spans

def attrs_per_byte(text, spans):
    b = len(text.encode("utf-8")); at = [([0.0, 0.0, 0.0, 0.0], 0, 0)] * b
    for s, e, box, lay, tag in spans:
        for i in range(s, e): at[i] = (box, lay, tag)
    return at

def main():
    r = random.Random(7); docs = []
    for k in range(240):
        L = KINDS[k % 3](r); text, spans = render(L); docs.append((f"document-{k+1:04d}", f"document-{KINDS[k%3].__name__}-{k//9}", text, spans))
    split = [_split(g, SEED) for _, g, _, _ in docs]
    tok = ByteBPETokenizer(); tok.train("\n".join(t for (_, _, t, _), s in zip(docs, split) if s == "train"), vocab_size=VOCAB)
    rows = []
    for i, g, text, spans in docs:
        ids = tok.encode(text)[:MAXLEN]; at = attrs_per_byte(text, spans); pos = 0; boxes, lays, tags = [], [], []
        for t in ids:
            n = len(tok.decode([t]).encode("utf-8")); seg = at[pos:pos + max(n, 1)]; pos += n
            pick = next((a for a in seg if a[1] != 0), seg[0] if seg else ([0.0] * 4, 0, 0))
            boxes.append(pick[0]); lays.append(pick[1]); tags.append(pick[2])
        rows.append({"id": i, "group_id": g, "text": text, "boxes": boxes, "layout_ids": lays, "tag_labels": tags})
    Path("data/authored/document/records.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(len(rows), "records, seed", SEED, {s: split.count(s) for s in ("train", "validation", "test")})
main()
