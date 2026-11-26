# Procedural score + SFX + narration mix for the Neural Pixel Lab film.
import json, sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

L = sys.argv[1]
SR = 44100
ev = json.load(open(f"{L}/events.json"))
TOTAL = ev["total"] + 0.5
n = int(TOTAL * SR)
rng = np.random.default_rng(3)

def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'low', fs=SR, output='sos'), x)
def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)
def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def midi(m): return 440.0 * 2 ** ((m - 69) / 12)
def add(buf, x, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf): return
    x = x[: len(buf) - i]
    buf[i:i + len(x)] += x * gain
def env_adsr(N, a, r):
    e = np.ones(N); A = int(a * SR); R = int(r * SR)
    if A: e[:A] = np.linspace(0, 1, A)
    if R: e[-R:] *= np.linspace(1, 0, R)
    return e
def saw(f, N, ph=0.0):
    t = np.arange(N) / SR
    return 2 * ((f * t + ph) % 1.0) - 1
def square(f, N, duty=0.5):
    t = np.arange(N) / SR
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)

scenes = {s["id"]: s for s in ev["scenes"]}
order = [s["id"] for s in ev["scenes"]]

# ---- harmony per scene (D minor world; failures drift darker) ----
CH0 = {
  "intro":    [38, 50, 57, 62, 65],          # Dm (add9 feel)
  "normal":   [38, 50, 57, 62, 65, 69],      # Dm
  "ascent":   [34, 46, 53, 58, 62, 65],      # Bb
  "linear":   [31, 43, 50, 55, 58, 62],      # Gm
  "sin":      [33, 45, 52, 57, 61, 64],      # A
  "noderiv":  [34, 46, 52, 58, 61, 64],      # Bb dim-ish (tense)
  "feedback": [29, 41, 48, 53, 57, 60, 65],  # F (lift)
  "adam":     [31, 43, 50, 55, 58, 63],      # Gm add b6 (dread)
  "momentum": [33, 45, 52, 57, 61, 64],      # A
  "outro":    [38, 50, 57, 62, 65, 69, 74],  # Dm -> resolve
}
CYC = [[38,50,57,62,65,69],[34,46,53,58,62,65],[31,43,50,55,58,62],[33,45,52,57,61,64]]
CH = {sid: CYC[i % 4] for i, sid in enumerate(order)}
CH["intro"] = CH0["intro"]; CH["outro"] = CH0["outro"]; CH["swap"] = CH0["feedback"]
pad = np.zeros(n); bass = np.zeros(n); arp = np.zeros(n); dark = np.zeros(n)
for sid in order:
    s = scenes[sid]; t0 = max(0, s["start"] - 0.6); dur = s["dur"] + 1.2
    N = int(dur * SR); notes = CH[sid]
    # pad: detuned saws, lowpassed, slow swell
    x = np.zeros(N)
    for m in notes[1:]:
        for d in (-0.08, 0.0, 0.09):
            x += saw(midi(m + d), N, rng.random())
    x = lp(x / (len(notes) * 3), 900 if sid not in ("div", "gsq") else 650)
    x *= env_adsr(N, 1.6, 1.4)
    add(pad, x, t0)
    # sub bass pulse (eighths at 84 bpm)
    tt = np.arange(N) / SR; beat = 60 / 84 / 2
    pulse = np.exp(-((tt % beat) / 0.18)) * 0.9 + 0.1
    b = np.sin(2 * np.pi * midi(notes[0]) * tt) + 0.3 * np.sin(2 * np.pi * midi(notes[0] + 12) * tt)
    b *= pulse * env_adsr(N, 1.0, 1.2)
    if sid == "intro": b *= np.clip((tt - 2) / 5, 0, 1)
    add(bass, b, t0)
    # chiptune arpeggio (16ths) for scenes that "work"
    if sid in ("normal", "swap", "outro", "xsq", "abs"):
        step = 60 / 84 / 4; k = 0; tt0 = 1.5
        seq = notes[2:] + notes[2:][::-1][1:-1]
        while tt0 < dur - 1.5:
            m = seq[k % len(seq)] + 12; L1 = int(step * 0.9 * SR)
            note = square(midi(m), L1, 0.25) * np.exp(-np.arange(L1) / SR / 0.07)
            add(arp, note, t0 + tt0, 0.5 if sid != "outro" else 0.65)
            tt0 += step; k += 1
    # dark noise bed for broken scenes
    if sid in ("div", "gsq", "flip", "plus"):
        nz = bp(rng.standard_normal(N), 180, 900) * env_adsr(N, 2.0, 1.5)
        add(dark, nz, t0, 0.5)

arp = lp(arp, 3800)
music = 0.55 * pad + 0.42 * bass + 0.08 * arp + 0.05 * dark

# ---- SFX ----
sfx = np.zeros(n)
# whoosh into every scene
for sid in order[1:]:
    s = scenes[sid]; N = int(1.4 * SR)
    nz = rng.standard_normal(N); e = np.linspace(0, 1, N) ** 3
    w = bp(nz, 300, 4000) * e
    add(sfx, w, s["start"] - 1.3, 0.22)
    # soft impact on cut
    N2 = int(0.9 * SR); tt = np.arange(N2) / SR
    hit = np.sin(2 * np.pi * (55 * np.exp(-tt * 3)) * tt) * np.exp(-tt * 5)
    add(sfx, hit, s["start"], 0.45)
# mutation glitch: bitcrushed burst + falling 8-bit chirp
for s in ev["scenes"]:
    if s["mutT"] is None: continue
    T = s["start"] + s["mutT"]; N = int(0.55 * SR); tt = np.arange(N) / SR
    nz = rng.standard_normal(N); crush = np.repeat(nz[::40], 40)[:N]
    chirp_f = 1400 * np.exp(-tt * 5) + 120
    chirp = np.sign(np.sin(2 * np.pi * np.cumsum(chirp_f) / SR))
    g = (0.5 * crush + 0.5 * chirp) * np.exp(-tt * 6)
    add(sfx, hp(g, 120), T, 0.30)
    # stutter
    for k in range(4):
        add(sfx, crush[:int(0.03 * SR)] * 0.6, T + 0.08 + k * 0.07, 0.25)
# boom (Adam explosion)
for s in ev["scenes"]:
    if s["boomT"] is None: continue
    T = s["start"] + s["boomT"]; N = int(3.0 * SR); tt = np.arange(N) / SR
    sub = np.sin(2 * np.pi * np.cumsum(70 * np.exp(-tt * 1.2) + 28) / SR) * np.exp(-tt * 1.3)
    nz = lp(rng.standard_normal(N), 2500) * np.exp(-tt * 2.2)
    add(sfx, 0.9 * sub + 0.35 * nz, T, 0.75)
# "it learns anyway" shimmer (feedback scene, sentence 3)
fb = scenes["swap"]; T = fb["start"] + fb["sent"][1][0] + 1.6
for i, m in enumerate([77, 81, 84, 89]):
    N = int(1.6 * SR); tt = np.arange(N) / SR
    add(sfx, np.sin(2 * np.pi * midi(m) * tt) * np.exp(-tt * 2.2), T + i * 0.09, 0.12)
# intro: pixels appearing (tiny blips)
it = scenes["intro"]
for i in range(256):
    x = np.sin(i * 12.9898 + 1 * 0) * 43758.5453
    appear = 0.3 + ((np.sin((i + 1) * 12.9898) * 43758.5453) % 1.0) * 3.5
    m = [74, 77, 81, 84, 86, 89][i % 6]; N = int(0.05 * SR)
    add(sfx, square(midi(m), N, 0.25) * np.exp(-np.arange(N) / SR / 0.015), it["start"] + appear, 0.035)
# training sonification: blips when the picture changes
fr = ev["frames"]; last = -10
PENTA = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84]
for f, (act, acc) in enumerate(fr):
    if act > 0.012 and f - last >= 2:
        last = f; m = PENTA[min(9, int(acc * 10))]; N = int(0.06 * SR)
        v = min(1.0, act * 12)
        add(sfx, square(midi(m), N, 0.25) * np.exp(-np.arange(N) / SR / 0.02), f / 30, 0.05 * v)

# ---- narration ----
voice = np.zeros(n)
for sid in order:
    s = scenes[sid]; v, sr = sf.read(f"{L}/audio/{sid}.wav")
    if v.ndim > 1: v = v.mean(1)
    if sr != SR: v = resample_poly(v, SR, sr)
    v = hp(v, 70)
    add(voice, v, s["start"] + s["lead"], 1.0)
# duck music under voice
envv = np.abs(voice); win = int(0.25 * SR)
envv = np.convolve(envv, np.ones(win) / win, mode="same")
duck = 1 - 0.45 * np.clip(envv / (envv.max() * 0.25 + 1e-9), 0, 1)
mix_music = (music * duck)

# stereo: slight width on pad/arp via delay
def stereo(x, d_ms):
    d = int(d_ms * SR / 1000); r = np.concatenate([np.zeros(d), x[:-d]]); return np.stack([x, r], 1)
mus_st = stereo(mix_music, 11) * 0.9
sfx_st = stereo(sfx, 4)
voc_st = np.stack([voice, voice], 1)
out = 0.42 * mus_st + 0.75 * sfx_st + 1.0 * voc_st
# fade end
fe = int(2.5 * SR); out[-fe:] *= np.linspace(1, 0, fe)[:, None]
out = np.tanh(out * 1.1) / np.tanh(1.1)
out *= 0.93 / np.abs(out).max()
sf.write(f"{L}/audio/mix.wav", out.astype(np.float32), SR)
print("mix", out.shape[0] / SR, "s")
