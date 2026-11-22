# -*- coding: utf-8 -*-
"""Procedurally generated, fully first-party VISION / IMAGE / 3D data (no external assets).
Run from repo root: python3 data/authored/make_synth_visual.py"""
import json, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

COLORS = {"kırmızı": (200, 40, 40), "mavi": (40, 80, 200), "yeşil": (40, 160, 70), "sarı": (230, 200, 40),
          "mor": (130, 60, 170), "turuncu": (235, 130, 30), "turkuaz": (30, 170, 180), "pembe": (230, 120, 170)}
BG = {"beyaz": (245, 245, 245), "siyah": (20, 20, 24), "gri": (140, 140, 146)}
SHAPES = ["daire", "kare", "üçgen", "dikdörtgen", "elmas", "artı"]
POS = {"ortada": (0.5, 0.5), "sol üstte": (0.27, 0.27), "sağ üstte": (0.73, 0.27),
       "sol altta": (0.27, 0.73), "sağ altta": (0.73, 0.73)}
ART = {"daire": "bir", "kare": "bir", "üçgen": "bir", "dikdörtgen": "bir", "elmas": "bir", "artı": "bir"}

def draw(shape, color, bg, pos, size, px, rng):
    img = Image.new("RGB", (px, px), BG[bg]); d = ImageDraw.Draw(img)
    cx, cy = POS[pos][0] * px, POS[pos][1] * px
    r = size * px
    c = COLORS[color]
    if shape == "daire": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "kare": d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "dikdörtgen": d.rectangle([cx - r * 1.5, cy - r * .8, cx + r * 1.5, cy + r * .8], fill=c)
    elif shape == "üçgen": d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=c)
    elif shape == "elmas": d.polygon([(cx, cy - r * 1.2), (cx - r, cy), (cx, cy + r * 1.2), (cx + r, cy)], fill=c)
    elif shape == "artı":
        t = r * .35; d.rectangle([cx - r, cy - t, cx + r, cy + t], fill=c); d.rectangle([cx - t, cy - r, cx + t, cy + r], fill=c)
    return img

def make_images(model, n, px, seed, size):
    rng = np.random.default_rng(seed); root = Path(f"data/authored/{model}"); (root / "images").mkdir(parents=True, exist_ok=True)
    combos = [(s, c, b, p) for s in SHAPES for c in COLORS for b in BG for p in POS]
    pick = rng.permutation(len(combos))[:n]; rows = []
    for k, ix in enumerate(pick, 1):
        s, c, b, p = combos[ix]
        img = draw(s, c, b, p, size, px, rng)
        name = f"{model}-{k:04d}.png"; img.save(root / "images" / name, optimize=True)
        cap = f"{b.capitalize()} zemin üzerinde {p} {c} {ART[s]} {s}."
        rows.append({"id": f"{model}-{k:04d}", "group_id": f"{model}-{s}-{c}", "image_path": f"images/{name}", "caption": cap})
    (root / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)

# ---------- 3D ----------
def norm(p):
    p = p - p.mean(0); return (p / (np.abs(p).max() + 1e-9) * 0.95).astype(np.float32)
def sphere(rng, n, sx, sy, sz):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); return v * [sx, sy, sz]
def cube(rng, n, sx, sy, sz):
    p = rng.uniform(-1, 1, (n, 3)); ax = rng.integers(0, 3, n); sg = rng.choice([-1, 1], n); p[np.arange(n), ax] = sg; return p * [sx, sy, sz]
def cylinder(rng, n, sx, sy, sz):
    t = rng.uniform(0, 2 * math.pi, n); z = rng.uniform(-1, 1, n); cap = rng.random(n) < .25
    r = np.where(cap, np.sqrt(rng.random(n)), 1.0); z = np.where(cap, rng.choice([-1, 1], n), z)
    return np.stack([r * np.cos(t) * sx, r * np.sin(t) * sy, z * sz], 1)
def cone(rng, n, sx, sy, sz):
    t = rng.uniform(0, 2 * math.pi, n); h = rng.random(n); r = (1 - h)
    return np.stack([r * np.cos(t) * sx, r * np.sin(t) * sy, (h * 2 - 1) * sz], 1)
def torus(rng, n, sx, sy, sz):
    u = rng.uniform(0, 2 * math.pi, n); v = rng.uniform(0, 2 * math.pi, n); R, r = 1.0, 0.35
    return np.stack([(R + r * np.cos(v)) * np.cos(u) * sx, (R + r * np.cos(v)) * np.sin(u) * sy, r * np.sin(v) * sz], 1)
def plane(rng, n, sx, sy, sz):
    p = rng.uniform(-1, 1, (n, 3)); p[:, 2] = rng.normal(0, .01, n); return p * [sx, sy, 1]
def pyramid(rng, n, sx, sy, sz):
    h = rng.random(n); a = rng.uniform(-1, 1, n); b = rng.uniform(-1, 1, n); s = 1 - h
    side = rng.random(n) < .7
    x = np.where(side, np.where(rng.random(n) < .5, a * s, np.sign(a) * s), a * s)
    y = np.where(side, np.where(rng.random(n) < .5, np.sign(b) * s, b * s), b * s)
    return np.stack([x * sx, y * sy, (h * 2 - 1) * sz], 1)
SH3 = {"küre": sphere, "küp": cube, "silindir": cylinder, "koni": cone, "halka": torus, "levha": plane, "piramit": pyramid}
PROP = {"normal": (1, 1, 1), "uzun": (1, 1, 1.9), "geniş": (1.9, 1.9, 1), "ince": (1, 1, 0.45), "yassı": (1.8, 1.0, 0.6)}

def make_3d(n, seed):
    rng = np.random.default_rng(seed); root = Path("data/authored/3d"); (root / "point_clouds").mkdir(parents=True, exist_ok=True)
    combos = [(s, q) for s in SH3 for q in PROP]; rows = []
    for k in range(n):
        s, q = combos[k % len(combos)]
        p = norm(SH3[s](rng, 1024, *PROP[q]))
        name = f"3d-{k+1:04d}.npy"; np.save(root / "point_clouds" / name, p)
        text = f"{q.capitalize()} bir {s}." if q != "normal" else f"Bir {s}."
        rows.append({"id": f"3d-{k+1:04d}", "group_id": f"3d-{s}-{q}-{k // len(combos)}", "point_path": f"point_clouds/{name}", "text": text})
    (root / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)

if __name__ == "__main__":
    print("vision", make_images("vision", 360, 224, 11, 0.16))
    print("image", make_images("image", 360, 64, 23, 0.2))
    print("3d", make_3d(210, 5))
