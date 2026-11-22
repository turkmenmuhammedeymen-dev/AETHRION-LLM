# -*- coding: utf-8 -*-
"""Procedural formant 'speech' for ASR / TTS / VOCODER. No human speaker is recorded or imitated: every clip is
synthesised from rules written for this repo. Transcript, per-token durations and waveform come from one timeline.
Run from repo root: PYTHONPATH=. python3 data/authored/make_synth_audio.py [seed]"""
import json, sys, wave, random
import numpy as np
from pathlib import Path
from aethrion.specialist_prepare import _split
from aethrion.tokenizer import ByteBPETokenizer
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SR, HOP = 16000, 160
V = {"a": (800, 1300), "e": (550, 1800), "ı": (350, 1300), "i": (300, 2200), "o": (500, 900), "ö": (500, 1500), "u": (320, 800), "ü": (320, 1600)}
FRIC = {"s": (6000, 1800), "ş": (3500, 1200), "f": (5000, 3000), "h": (1800, 1500), "z": (5500, 1800), "j": (3500, 1200), "v": (900, 700)}
STOP = {"p": 800, "t": 4000, "k": 2200, "ç": 3500, "b": 700, "d": 3000, "g": 1800, "c": 3200}
SONOR = {"m": (250, 1100), "n": (300, 1500), "l": (400, 1500), "r": (450, 1300), "y": (300, 2200), "ğ": (400, 1100)}
VOICES = [(110, 1.0), (135, 0.97), (165, 1.05), (210, 1.12), (250, 1.08)]
WORDS = "merhaba bugün hava güzel çok teşekkür ederim lütfen yardım evet hayır tamam nasılsın iyiyim kitap okumak yazı kalem masa su ekmek sabah akşam gece okul ders sınıf arkadaş aile müzik şarkı yol araba otobüs kahve çay elma portakal bahçe çiçek güneş yağmur kar deniz dağ şehir köy sayı bir iki üç dört beş altı yedi sekiz dokuz on".split()

def seg(ch, f0, fs, rng):
    n = lambda ms: int(SR * ms / 1000); t = lambda m: np.arange(m) / SR
    if ch == " ": return np.zeros(n(70))
    if ch in V:
        m = n(120); x = t(m); f1, f2 = V[ch][0] * fs, V[ch][1] * fs
        src = sum(np.sin(2 * np.pi * f0 * h * x) / h for h in range(1, 18))
        w = np.sin(2 * np.pi * f1 * x) * .6 + np.sin(2 * np.pi * f2 * x) * .35
        y = src * (0.35 + 0.65 * np.abs(w)) * np.sin(np.pi * np.clip(x / x[-1], 0, 1)) ** .5
        return y * 0.5
    if ch in SONOR:
        m = n(90); x = t(m); f1, f2 = SONOR[ch][0] * fs, SONOR[ch][1] * fs
        src = sum(np.sin(2 * np.pi * f0 * h * x) / h for h in range(1, 10)); w = .5 * np.sin(2 * np.pi * f1 * x) + .2 * np.sin(2 * np.pi * f2 * x)
        return src * (0.3 + 0.4 * np.abs(w)) * np.hanning(m) * 0.3
    if ch in FRIC:
        m = n(95); x = t(m); fc, bw = FRIC[ch]; noise = rng.normal(size=m)
        y = noise * np.sin(2 * np.pi * fc * x) * np.hanning(m) * 0.25
        if ch in "zjv": y += 0.15 * np.sin(2 * np.pi * f0 * x) * np.hanning(m)
        return y
    if ch in STOP:
        m = n(80); y = np.zeros(m); b = n(25); x = t(b); fc = STOP[ch]
        y[n(40):n(40) + b] = rng.normal(size=b) * np.sin(2 * np.pi * fc * x) * np.hanning(b) * 0.4
        if ch in "bdgc": y[n(40):n(40) + b] += 0.2 * np.sin(2 * np.pi * f0 * x) * np.hanning(b)
        return y
    return np.zeros(n(30))

def synth(text, voice, rng):
    f0, fs = VOICES[voice]; parts, bounds = [], []
    for ch in text:
        s = seg(ch, f0 * (1 + rng.normal(0, .01)), fs, rng); parts.append(s)
    wav = np.concatenate([np.zeros(int(SR * .08))] + parts + [np.zeros(int(SR * .08))])
    starts = np.cumsum([int(SR * .08)] + [len(p) for p in parts])[:-1]
    wav = wav / (np.abs(wav).max() + 1e-9) * 0.7
    return wav.astype(np.float32), starts, len(wav)

def char_frames(starts, n_samples):
    frames = n_samples // HOP + 1; cf = [0] * len(starts)
    for i in range(frames):
        pos = i * HOP; c = max(0, int(np.searchsorted(starts, pos, side="right") - 1)); cf[c] += 1
    cf[0] += 0
    return cf, frames

def main():
    r = random.Random(3); rng = np.random.default_rng(3)
    phrases, seen = [], set()
    while len(phrases) < 160:
        p = " ".join(r.choice(WORDS) for _ in range(r.randint(1, 2)))
        if p not in seen: seen.add(p); phrases.append(p)
    split = [_split(f"audio-phrase-{k}", SEED) for k in range(len(phrases))]
    tok = ByteBPETokenizer(); tok.train("\n".join(p for p, s in zip(phrases, split) if s == "train"), vocab_size=2048)
    out = Path("data/authored/audio"); (out / "wav").mkdir(parents=True, exist_ok=True)
    asr, tts, voc = [], [], []
    for k, ph in enumerate(phrases):
        wav, starts, n = synth(ph, k % len(VOICES), rng)
        name = f"clip-{k+1:04d}.wav"
        with wave.open(str(out / "wav" / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((wav * 32767).astype("<i2").tobytes())
        cf, frames = char_frames(starts, n)
        # per-byte frame shares, then per-token sums
        byte_frames = []
        for ch, f in zip(ph, cf):
            nb = len(ch.encode("utf-8")); base = f // nb; rem = f - base * nb
            byte_frames += [base + (1 if j < rem else 0) for j in range(nb)]
        ids = tok.encode(ph); dur, pos = [], 0
        for t in ids:
            nb = len(tok.decode([t]).encode("utf-8")); dur.append(sum(byte_frames[pos:pos + nb])); pos += nb
        assert pos == len(byte_frames) and sum(dur) == frames and min(dur) >= 1, (ph, dur, frames)
        base = {"id": f"audio-{k+1:04d}", "group_id": f"audio-phrase-{k}", "speaker_id": f"synthetic-voice-{k % len(VOICES)}", "wav_path": f"wav/{name}"}
        asr.append({**base, "transcript": ph}); tts.append({**base, "transcript": ph, "durations": dur}); voc.append(base)
    for m, rows in (("asr", asr), ("tts", tts), ("vocoder", voc)):
        (out / f"{m}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(len(asr), "clips; splits", {s: split.count(s) for s in ("train", "validation", "test")})
main()
