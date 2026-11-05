"""Procedural audio for the Emberdeep renderer.

Every sound effect is synthesised with numpy (noise, additive/modal synthesis, FFT filtering and a
cheap convolution reverb), cached as ``sfx_v*.npz`` in :func:`emberdeep.render.assets.cache_dir` and
played through ``pygame.mixer``.  The engine is *fail-safe*: when no audio device / mixer is
available (headless CI, dummy drivers refusing, ``enabled=False``) every method is a silent no-op
and nothing ever raises.

    audio = AudioEngine(volume=0.7)
    audio.on_event(event_dict, pan=-0.3)      # map an engine event to sounds
    audio.update(dt)                          # per frame: delayed sounds + music crossfade
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SR = 44100
CACHE_VERSION = 3

# EventType values (core/events.py) - duplicated as ints so the audio module needs no engine import.
try:  # pragma: no cover - trivial
    from emberdeep.core.events import EventType as ET
except Exception:  # noqa: BLE001
    ET = None  # type: ignore[assignment]


# =============================================================================== DSP helpers
def _n(dur: float) -> int:
    return max(1, int(dur * SR))


def _t(dur: float) -> np.ndarray:
    return np.arange(_n(dur), dtype=np.float64) / SR


def _env_exp(n: int, tau: float, attack: float = 0.002) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.exp(-t / max(1e-4, tau))
    a = max(1, int(attack * SR))
    e[:a] *= np.linspace(0, 1, a)
    return e


def _fft_filter(x: np.ndarray, lo: float | None, hi: float | None, order: int = 3, circular: bool = False) -> np.ndarray:
    """Zero-phase Butterworth-magnitude band filter via FFT. ``circular`` keeps loops seamless."""
    n = len(x)
    N = n if circular else int(2 ** np.ceil(np.log2(n * 2)))
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1.0 / SR)
    g = np.ones_like(f)
    if hi:
        g *= 1.0 / np.sqrt(1.0 + (f / hi) ** (2 * order))
    if lo:
        ff = np.maximum(f, 1e-3)
        g *= 1.0 / np.sqrt(1.0 + (lo / ff) ** (2 * order))
    y = np.fft.irfft(X * g, N)
    return y[:n]


def _reverb(x: np.ndarray, rt: float = 0.5, wet: float = 0.25, rng: np.random.Generator | None = None,
            circular: bool = False, damp: float = 5000.0) -> np.ndarray:
    rng = rng or np.random.default_rng(11)
    ir_n = _n(rt)
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR * 6.9 / rt)
    ir = _fft_filter(ir, 120.0, damp, 2)
    ir[0] = 0.0
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    n = len(x)
    if circular:
        N = n
        irp = np.zeros(N)
        irp[:min(N, ir_n)] = ir[:N]
        y = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(irp), N)
        return x * (1 - wet) + y * wet * 1.8
    N = int(2 ** np.ceil(np.log2(n + ir_n)))
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:n + ir_n]
    out = np.zeros(n + ir_n)
    out[:n] = x * (1 - wet)
    out += y * wet * 1.8
    return out


def _norm(x: np.ndarray, peak: float = 0.85) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return x * (peak / m)


def _fade(x: np.ndarray, ms_in: float = 2.0, ms_out: float = 12.0) -> np.ndarray:
    x = x.copy()
    a, b = int(ms_in * SR / 1000), int(ms_out * SR / 1000)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def _place(dst: np.ndarray, src: np.ndarray, at: float, gain: float = 1.0, wrap: bool = False) -> None:
    i = int(at * SR)
    n = len(src)
    if wrap:
        idx = (np.arange(n) + i) % len(dst)
        np.add.at(dst, idx, src * gain)
        return
    if i >= len(dst):
        return
    j = min(len(dst), i + n)
    dst[i:j] += src[:j - i] * gain


def _sine(freq, dur: float, tau: float, phase: float = 0.0) -> np.ndarray:
    n = _n(dur)
    if np.isscalar(freq):
        ph = 2 * np.pi * freq * np.arange(n) / SR
    else:
        ph = 2 * np.pi * np.cumsum(freq[:n]) / SR
    return np.sin(ph + phase) * _env_exp(n, tau)


def _sweep(f0: float, f1: float, dur: float, tau: float) -> np.ndarray:
    n = _n(dur)
    f = f1 + (f0 - f1) * np.exp(-np.arange(n) / SR / (dur * 0.25))
    return _sine(f, dur, tau)


def _bell(freq: float, dur: float, tau: float, ratios=(1.0, 2.76, 5.4, 8.93), gains=(1.0, 0.5, 0.25, 0.12)) -> np.ndarray:
    n = _n(dur)
    out = np.zeros(n)
    for r, g in zip(ratios, gains):
        out += g * _sine(freq * r, dur, tau / (0.6 + 0.4 * r))
    return out


def _saw_band(freq: float, dur: float, harmonics: int = 14, vib: float = 0.0, vib_hz: float = 5.5) -> np.ndarray:
    n = _n(dur)
    t = np.arange(n) / SR
    f = freq * (1 + vib * np.sin(2 * np.pi * vib_hz * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        out += np.sin(h * ph) / h
    return out


# =============================================================================== sound recipes
def _s_sword_swing(r):
    n = _n(0.38)
    nz = r.standard_normal(n)
    lo = _fft_filter(nz, 300, 1400)
    hi = _fft_filter(nz, 1800, 7000)
    t = np.arange(n) / SR
    pos = np.clip(t / 0.30, 0, 1)
    mix = pos ** 1.5
    env = np.sin(np.pi * np.clip(t / 0.34, 0, 1)) ** 2.2
    return _fade(_norm((lo * (1 - mix) + hi * mix * 1.2) * env, 0.6))


def _s_clang(r):
    dur = 1.1
    out = np.zeros(_n(dur))
    base = r.uniform(780, 900)
    for ratio, g, tau in ((1.0, 1.0, 0.30), (1.58, 0.7, 0.22), (2.41, 0.6, 0.18), (3.63, 0.45, 0.12), (5.11, 0.35, 0.08),
                          (7.3, 0.2, 0.05)):
        out += g * _sine(base * ratio * r.uniform(0.99, 1.01), dur, tau)
    click = _fft_filter(r.standard_normal(_n(dur)), 2500, 9000) * _env_exp(_n(dur), 0.006)
    out += click * 0.9
    return _fade(_norm(_reverb(out, 0.35, 0.2, r), 0.75))


def _s_flesh_hit(r):
    n = _n(0.34)
    thump = _sweep(150, 52, 0.34, 0.07) * 1.0
    body = _fft_filter(r.standard_normal(n), 80, 900) * _env_exp(n, 0.05)
    slap = _fft_filter(r.standard_normal(n), 900, 3500) * _env_exp(n, 0.012) * 0.35
    return _fade(_norm(thump + body * 1.2 + slap, 0.8))


def _s_bow_twang(r):
    dur = 0.55
    n = _n(dur)
    t = np.arange(n) / SR
    f0 = 196.0
    f = f0 * (1 + 0.04 * np.exp(-t / 0.03))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(n)
    for h, g in ((1, 1.0), (2, 0.6), (3, 0.45), (4, 0.3), (5, 0.2), (7, 0.1)):
        out += g * np.sin(h * ph) * np.exp(-t / (0.16 / (0.6 + 0.4 * h)))
    rel = _fft_filter(r.standard_normal(n), 1500, 6000) * _env_exp(n, 0.015) * 0.4
    return _fade(_norm(out + rel, 0.7))


def _s_arrow_impact(r):
    n = _n(0.28)
    thud = _sweep(210, 80, 0.28, 0.05)
    tick = _fft_filter(r.standard_normal(n), 1200, 4200) * _env_exp(n, 0.012)
    wood = _sine(420, 0.28, 0.03) * 0.3
    return _fade(_norm(thud + tick * 0.9 + wood, 0.75))


def _s_fire_whoosh(r):
    dur = 1.0
    n = _n(dur)
    t = np.arange(n) / SR
    nz = r.standard_normal(n)
    body = _fft_filter(nz, 120, 1400)
    env = np.minimum(1, t / 0.15) * np.exp(-np.maximum(0, t - 0.15) / 0.35)
    crackle = np.zeros(n)
    for _ in range(26):
        at = r.uniform(0.05, 0.9)
        k = _fft_filter(r.standard_normal(_n(0.03)), 1500, 7000) * _env_exp(_n(0.03), 0.006)
        _place(crackle, k, at, r.uniform(0.2, 0.9))
    return _fade(_norm(body * env + crackle * 0.45 * env, 0.75), 4, 40)


def _s_explosion(r):
    dur = 1.8
    n = _n(dur)
    boom = _sweep(110, 28, dur, 0.38) * 1.2
    rumble = _fft_filter(r.standard_normal(n), 30, 700) * _env_exp(n, 0.45)
    crack = _fft_filter(r.standard_normal(n), 400, 5000) * _env_exp(n, 0.06) * 0.9
    return _fade(_norm(_reverb(boom + rumble * 1.2 + crack, 0.8, 0.25, r), 0.85), 2, 60)


def _s_ice_crackle(r):
    dur = 1.0
    n = _n(dur)
    out = np.zeros(n)
    for _ in range(34):
        at = r.uniform(0, 0.8)
        f = r.uniform(2500, 7000)
        k = _sine(f, 0.06, 0.012) * r.uniform(0.2, 0.8)
        _place(out, k, at)
    glass = _bell(r.uniform(2200, 2600), dur, 0.25, (1, 1.51, 2.3), (0.5, 0.35, 0.2)) * 0.5
    nz = _fft_filter(r.standard_normal(n), 3500, 10000) * _env_exp(n, 0.2) * 0.3
    return _fade(_norm(_reverb(out + glass + nz, 0.4, 0.25, r), 0.7), 2, 40)


def _s_lightning_crack(r):
    dur = 0.9
    n = _n(dur)
    crack = r.standard_normal(n) * _env_exp(n, 0.02, 0.0005)
    crack = _fft_filter(crack, 300, 12000, 2)
    zap = _fft_filter(r.standard_normal(n), 2000, 9000) * np.abs(np.sin(2 * np.pi * 60 * np.arange(n) / SR)) * _env_exp(n, 0.12)
    rumble = _fft_filter(r.standard_normal(n), 40, 400) * _env_exp(n, 0.3) * 0.6
    return _fade(_norm(_reverb(crack * 1.2 + zap * 0.6 + rumble, 0.6, 0.25, r), 0.85), 1, 50)


def _s_thunder_boom(r):
    dur = 2.4
    n = _n(dur)
    t = np.arange(n) / SR
    rumble = _fft_filter(r.standard_normal(n), 25, 260)
    env = np.minimum(1, t / 0.12) * np.exp(-t / 0.9) * (1 + 0.5 * np.sin(2 * np.pi * 7 * t) * np.exp(-t / 0.8))
    sub = _sweep(70, 34, dur, 0.7)
    return _fade(_norm(_reverb(rumble * env * 1.5 + sub * 0.8, 1.2, 0.3, r, damp=1500), 0.85), 3, 100)


def _s_magic_chime(r):
    dur = 1.5
    out = np.zeros(_n(dur))
    notes = [880.0, 1318.5, 1760.0]
    for i, f in enumerate(notes):
        _place(out, _bell(f, 1.2, 0.5 + 0.1 * i, (1, 2.0, 3.01, 4.2), (1, 0.3, 0.15, 0.08)), 0.0 + i * 0.09, 0.6)
    return _fade(_norm(_reverb(out, 0.9, 0.3, r), 0.6), 2, 80)


def _s_sparkle(r):
    dur = 0.9
    out = np.zeros(_n(dur))
    for i in range(14):
        f = r.choice([2093, 2637, 3136, 3951, 4186, 5274])
        _place(out, _sine(f * r.uniform(0.995, 1.005), 0.2, 0.05), r.uniform(0, 0.6), r.uniform(0.2, 0.7))
    return _fade(_norm(_reverb(out, 0.5, 0.35, r), 0.5), 2, 50)


def _s_heal(r):
    dur = 1.5
    out = np.zeros(_n(dur))
    for i, f in enumerate([523.25, 659.25, 783.99, 1046.5, 1318.5]):
        tone = _sine(f, 0.9, 0.35) + 0.3 * _sine(f * 2, 0.9, 0.2)
        _place(out, tone, i * 0.075, 0.55)
    shimmer = _fft_filter(r.standard_normal(_n(dur)), 4000, 9000) * np.sin(np.pi * np.clip(np.arange(_n(dur)) / SR / dur, 0, 1)) * 0.06
    return _fade(_norm(_reverb(out + shimmer, 1.0, 0.35, r), 0.6), 2, 90)


def _s_necrotic(r):
    dur = 1.2
    n = _n(dur)
    t = np.arange(n) / SR
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    hiss = _fft_filter(r.standard_normal(n), 3000, 9000) * env * 0.7
    drone = (np.sin(2 * np.pi * 69 * t) + 0.7 * np.sin(2 * np.pi * 73.4 * t) + 0.4 * np.sin(2 * np.pi * 103 * t)) * env * 0.35
    return _fade(_norm(_reverb(hiss + drone, 0.7, 0.3, r), 0.6), 5, 80)


def _s_poison(r):
    dur = 1.0
    n = _n(dur)
    out = _fft_filter(r.standard_normal(n), 100, 600) * _env_exp(n, 0.4) * 0.25
    for _ in range(9):
        f0 = r.uniform(180, 380)
        pop = _sine(f0 * (1 + 2.0 * np.exp(-np.arange(_n(0.09)) / SR / 0.02)), 0.09, 0.03)
        _place(out, pop, r.uniform(0, 0.8), r.uniform(0.4, 1.0))
    return _fade(_norm(_reverb(out, 0.3, 0.2, r), 0.6), 2, 50)


def _s_footstep(r):
    n = _n(0.26)
    thud = _fft_filter(r.standard_normal(n), 60, 650) * _env_exp(n, 0.05)
    low = _sine(r.uniform(80, 105), 0.26, 0.06) * 0.8
    scuff = _fft_filter(r.standard_normal(n), 1200, 5000) * _env_exp(n, 0.02) * 0.25
    return _fade(_norm(thud + low + scuff, 0.5))


def _s_door_creak(r):
    dur = 1.4
    n = _n(dur)
    t = np.arange(n) / SR
    f = 150 + 90 * np.sin(np.pi * np.clip(t / dur, 0, 1)) + 14 * np.sin(2 * np.pi * 9 * t) + r.uniform(-8, 8)
    ph = 2 * np.pi * np.cumsum(f) / SR
    wave = np.zeros(n)
    for h in range(1, 12):
        wave += np.sin(h * ph) / h ** 1.1
    wave = _fft_filter(wave, 250, 1800)
    noise = _fft_filter(r.standard_normal(n), 400, 2500) * 0.25
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.8 * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))
    return _fade(_norm(_reverb((wave + noise) * env, 0.4, 0.2, r), 0.5), 20, 100)


def _s_chest_open(r):
    out = np.zeros(_n(1.0))
    latch = _fft_filter(r.standard_normal(_n(0.05)), 1800, 6000) * _env_exp(_n(0.05), 0.008)
    _place(out, latch, 0.0, 0.9)
    creak = _s_door_creak(r)[:_n(0.55)] * 0.7
    _place(out, creak, 0.06, 0.7)
    _place(out, _s_flesh_hit(r)[:_n(0.2)], 0.58, 0.55)
    _place(out, _bell(1900, 0.4, 0.15, (1, 2.3), (1, 0.4)), 0.62, 0.25)
    return _fade(_norm(out, 0.6), 2, 60)


def _s_coin(r):
    dur = 1.0
    out = np.zeros(_n(dur))
    for i in range(7):
        f = r.uniform(2300, 3600)
        c = _sine(f, 0.4, 0.08) + 0.6 * _sine(f * 1.51, 0.4, 0.05)
        _place(out, c, i * r.uniform(0.04, 0.08), r.uniform(0.35, 0.8))
    return _fade(_norm(_reverb(out, 0.3, 0.2, r), 0.55), 1, 50)


def _s_death_thud(r):
    dur = 1.0
    n = _n(dur)
    boom = _sweep(95, 34, dur, 0.22) * 1.1
    body = _fft_filter(r.standard_normal(n), 50, 500) * _env_exp(n, 0.1)
    rustle = _fft_filter(r.standard_normal(n), 700, 3000) * _env_exp(n, 0.2) * 0.12
    return _fade(_norm(_reverb(boom + body + rustle, 0.5, 0.2, r), 0.85), 2, 60)


def _s_dice(r):
    dur = 0.85
    out = np.zeros(_n(dur))
    t = 0.0
    gap = 0.012
    while t < 0.7:
        f = r.uniform(1800, 5000)
        k = _fft_filter(r.standard_normal(_n(0.02)), f * 0.6, f * 1.5) * _env_exp(_n(0.02), 0.004)
        _place(out, k, t, r.uniform(0.3, 1.0) * (1 - t / 0.9))
        t += gap * r.uniform(0.6, 1.6)
        gap *= 1.07
    return _fade(_norm(out, 0.5), 1, 60)


def _s_ui_click(r):
    n = _n(0.09)
    tick = _sine(1700, 0.09, 0.012) * 0.8 + _fft_filter(r.standard_normal(n), 2000, 7000) * _env_exp(n, 0.004) * 0.35
    return _fade(_norm(tick, 0.4), 0.5, 10)


def _s_ui_hover(r):
    return _fade(_norm(_sine(1250, 0.06, 0.012) + 0.4 * _sine(2500, 0.06, 0.008), 0.18), 0.5, 10)


def _horn(freqs, dur: float, attack: float, r, bad: bool = False, vib: float = 0.004) -> np.ndarray:
    n = _n(dur)
    t = np.arange(n) / SR
    env = np.minimum(1, t / attack) ** 1.4 * np.exp(-np.maximum(0, t - dur * 0.55) / (dur * 0.22))
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        w = _saw_band(f, dur, 16, vib, 5.2 + i * 0.3)
        out += w
    # brass-like cutoff that opens with the swell
    soft = _fft_filter(out, 80, 900 if not bad else 700, 2)
    bright = _fft_filter(out, 80, 3200, 2)
    mix = np.clip(t / (attack * 1.4), 0, 1) ** 1.5
    sig = (soft * (1 - mix) + bright * mix) * env
    return _fade(_norm(_reverb(sig, 1.1, 0.28, r), 0.7), 8, 120)


def _s_horn_swell(r):
    return _horn([233.1, 349.2, 466.2], 2.0, 0.55, r)


def _s_horn_bad(r):
    return _horn([196.0, 277.2, 392.0], 1.6, 0.35, r, bad=True, vib=0.012)


def _s_horn_win(r):
    out = np.zeros(_n(3.0))
    for i, (f, d) in enumerate(((261.6, 0.5), (329.6, 0.5), (392.0, 0.5), (523.3, 1.6))):
        _place(out, _horn([f, f * 1.5], d + 0.3, 0.12, r), i * 0.45, 0.8)
    return _fade(_norm(out, 0.7), 5, 150)


def _s_horn_lose(r):
    out = np.zeros(_n(3.0))
    for i, (f, d) in enumerate(((220.0, 0.8), (207.7, 0.8), (174.6, 1.8))):
        _place(out, _horn([f, f * 1.19], d + 0.4, 0.25, r, bad=True), i * 0.8, 0.8)
    return _fade(_norm(out, 0.7), 5, 150)


SFX: dict[str, Callable] = {
    "sword_swing": _s_sword_swing, "clang": _s_clang, "flesh_hit": _s_flesh_hit, "bow_twang": _s_bow_twang,
    "arrow_impact": _s_arrow_impact, "fire_whoosh": _s_fire_whoosh, "explosion": _s_explosion,
    "ice_crackle": _s_ice_crackle, "lightning_crack": _s_lightning_crack, "thunder_boom": _s_thunder_boom,
    "magic_chime": _s_magic_chime, "sparkle": _s_sparkle, "heal_shimmer": _s_heal, "necrotic_hiss": _s_necrotic,
    "poison_bubble": _s_poison, "footstep_a": _s_footstep, "footstep_b": _s_footstep, "door_creak": _s_door_creak,
    "chest_open": _s_chest_open, "coin_jingle": _s_coin, "death_thud": _s_death_thud, "dice_rattle": _s_dice,
    "ui_click": _s_ui_click, "ui_hover": _s_ui_hover, "horn_swell": _s_horn_swell, "horn_bad": _s_horn_bad,
    "horn_win": _s_horn_win, "horn_lose": _s_horn_lose,
}
BASE_GAIN = {"sword_swing": 0.6, "clang": 0.55, "flesh_hit": 0.8, "bow_twang": 0.6, "arrow_impact": 0.7,
             "fire_whoosh": 0.6, "explosion": 0.85, "ice_crackle": 0.55, "lightning_crack": 0.7, "thunder_boom": 0.8,
             "magic_chime": 0.45, "sparkle": 0.4, "heal_shimmer": 0.5, "necrotic_hiss": 0.5, "poison_bubble": 0.55,
             "footstep_a": 0.4, "footstep_b": 0.4, "door_creak": 0.55, "chest_open": 0.6, "coin_jingle": 0.5,
             "death_thud": 0.85, "dice_rattle": 0.45, "ui_click": 0.5, "ui_hover": 0.3, "horn_swell": 0.55,
             "horn_bad": 0.55, "horn_win": 0.6, "horn_lose": 0.6}
VARIANT_SPEEDS = (0.94, 1.0, 1.07)
NO_VARIANTS = {"horn_swell", "horn_bad", "horn_win", "horn_lose", "thunder_boom", "ui_click", "ui_hover"}


def make_ambient(r: np.random.Generator, T: float = 16.0) -> np.ndarray:
    """Seamless dungeon ambience: low drone (integer cycles per loop) + rumble + water drips."""
    n = int(T * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, (f, g) in enumerate(((55.0, 0.5), (82.5, 0.28), (110.0, 0.2), (165.0, 0.1), (41.25, 0.3))):
        f = round(f * T) / T                      # whole cycles per loop
        lfo = 0.65 + 0.35 * np.sin(2 * np.pi * (k + 1) / T * t + k)
        out += g * np.sin(2 * np.pi * f * t + k * 1.3) * lfo
    rumble = _fft_filter(r.standard_normal(n), 25, 220, 2, circular=True)
    out += _norm(rumble, 0.35) * (0.7 + 0.3 * np.sin(2 * np.pi * 2 / T * t))
    air = _fft_filter(r.standard_normal(n), 300, 1500, 2, circular=True)
    out += _norm(air, 0.05)
    drips = np.zeros(n)
    for _ in range(7):
        at = r.uniform(0.5, T - 1.0)
        f = r.uniform(900, 2200)
        d = _sine(f * (1 + 0.25 * np.exp(-np.arange(_n(0.5)) / SR / 0.02)), 0.5, 0.05)
        d += 0.4 * _sine(f * 2.4, 0.5, 0.02)
        _place(drips, d, at, r.uniform(0.25, 0.55), wrap=True)
    drips = _reverb(drips, 1.4, 0.55, r, circular=True, damp=4000)
    out += drips * 0.5
    return _norm(out, 0.8)


def make_combat_loop(r: np.random.Generator, bpm: float = 104.0, beats: int = 8) -> np.ndarray:
    """Seamless taiko-ish drum pulse with a low brass drone."""
    beat = 60.0 / bpm
    T = beat * beats
    n = int(T * SR)
    out = np.zeros(n)
    kick = _sweep(135, 44, 0.5, 0.14)
    kick = kick + _fft_filter(r.standard_normal(_n(0.5)), 60, 400) * _env_exp(_n(0.5), 0.03) * 0.5
    tom = _sweep(190, 80, 0.5, 0.12) + 0.4 * _fft_filter(r.standard_normal(_n(0.5)), 120, 800) * _env_exp(_n(0.5), 0.02)
    for b in range(beats):
        if b % 2 == 0:
            _place(out, kick, b * beat, 1.0, wrap=True)
        if b in (3, 7):
            _place(out, tom, b * beat + beat * 0.5, 0.7, wrap=True)
            _place(out, tom, b * beat + beat * 0.75, 0.55, wrap=True)
        if b in (1, 5):
            _place(out, tom * 0.8, b * beat + beat * 0.5, 0.55, wrap=True)
    t = np.arange(n) / SR
    f1 = round(55.0 * T) / T
    drone = (np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f1 * 1.5 * t + 0.4)) * (0.55 + 0.45 * np.sin(2 * np.pi * 2 / T * t))
    out = out + 0.22 * drone
    out = _reverb(out, 0.5, 0.18, r, circular=True, damp=3000)
    return _norm(out, 0.8)


# =============================================================================== engine
def _to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def synth_bank() -> dict[str, np.ndarray]:
    """Synthesise every sound (mono int16). Deterministic."""
    bank: dict[str, np.ndarray] = {}
    for i, (name, fn) in enumerate(SFX.items()):
        r = np.random.default_rng(1000 + i * 7)
        bank[name] = _to_int16(_norm(fn(r), 0.9))
    bank["ambient"] = _to_int16(make_ambient(np.random.default_rng(77)))
    bank["combat"] = _to_int16(make_combat_loop(np.random.default_rng(78)))
    return bank


def _cache_path():
    from emberdeep.render import assets
    return assets.cache_dir() / f"sfx_v{CACHE_VERSION}.npz"


def load_or_build_bank() -> dict[str, np.ndarray]:
    try:
        p = _cache_path()
        if p.exists():
            with np.load(p) as z:
                bank = {k: z[k] for k in z.files}
            if set(bank) >= set(SFX) | {"ambient", "combat"}:
                return bank
    except Exception:  # noqa: BLE001
        log.debug("audio cache unreadable", exc_info=True)
    bank = synth_bank()
    try:
        np.savez_compressed(_cache_path(), **bank)
        assets.prune_cache("sfx_v", _cache_path().name)
    except Exception:  # noqa: BLE001
        log.debug("audio cache not writable", exc_info=True)
    return bank


def _resample(x: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6:
        return x
    n = int(len(x) / speed)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


# event -> sound keyword tables (spell vfx keys come from render.api.VFX_KEYS)
_FIRE = {"fire_bolt", "fireball", "burning_hands", "scorching_ray", "breath_fire", "produce_flame", "flaming_sphere",
         "wall_of_fire", "throw_flask", "trap_fire", "heat_metal"}
_ICE = {"ray_of_frost", "cone_of_cold", "breath_cold", "sleet_storm", "frost"}
_LIGHTNING = {"lightning_bolt", "call_lightning", "shocking_grasp", "breath_lightning"}
_THUNDER = {"thunderwave", "shatter"}
_HEAL = {"healing", "mass_healing", "revive", "lay_on_hands", "second_wind", "aid", "cure_wounds", "potion", "heroism",
         "stabilize", "level_up"}
_NECRO = {"chill_touch", "inflict_wounds", "vampiric_touch", "ray_of_enfeeblement", "hex_curse", "bane", "death"}
_POISON = {"acid_splash", "acid_arrow", "poison_spray", "stinking_cloud", "breath_acid", "breath_poison", "spit"}
_PHYS = {1, 7, 11}


class AudioEngine:
    def __init__(self, enabled: bool = True, volume: float = 0.7, lazy: bool = False) -> None:
        self.enabled = bool(enabled)
        self.volume = float(max(0.0, min(1.0, volume)))
        self.muted = False
        self.ok = False
        self._pg = None
        self._sounds: dict[str, list] = {}
        self._music: dict[str, object] = {}
        self._music_ch: dict[str, object] = {}
        self._music_vol = {"ambient": 0.0, "combat": 0.0}
        self._combat = False
        self._queue: list[tuple[float, str, float, float]] = []
        self._clock = 0.0
        self._last_play: dict[str, float] = {}
        self._rng = random.Random(5)
        self._last_attack: dict[int, tuple[float, bool]] = {}
        if not self.enabled:
            return
        try:
            self._init()
        except Exception:  # noqa: BLE001
            log.info("audio disabled (mixer unavailable)", exc_info=True)
            self.ok = False

    # ------------------------------------------------------------------ setup
    def _init(self) -> None:
        import pygame
        self._pg = pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512)
        info = pygame.mixer.get_init()
        if not info:
            raise RuntimeError("mixer failed to initialise")
        freq, _, chans = info
        try:
            pygame.mixer.set_num_channels(28)
            pygame.mixer.set_reserved(2)
        except Exception:  # noqa: BLE001
            pass
        bank = load_or_build_bank()
        ratio = freq / SR

        def mk(mono: np.ndarray):
            if abs(ratio - 1) > 1e-3:
                mono = _resample(mono, 1.0 / ratio)
            if chans == 1:
                arr = np.ascontiguousarray(mono)
            else:
                arr = np.ascontiguousarray(np.repeat(mono[:, None], chans, axis=1))
            try:
                return pygame.sndarray.make_sound(arr)
            except Exception:  # noqa: BLE001
                return pygame.mixer.Sound(buffer=arr.tobytes())

        for name in SFX:
            mono = bank[name]
            speeds = (1.0,) if name in NO_VARIANTS else VARIANT_SPEEDS
            self._sounds[name] = [mk(_resample(mono, s)) for s in speeds]
        self._music["ambient"] = mk(bank["ambient"])
        self._music["combat"] = mk(bank["combat"])
        for i, key in enumerate(("ambient", "combat")):
            ch = pygame.mixer.Channel(i)
            ch.set_volume(0.0)
            ch.play(self._music[key], loops=-1)
            self._music_ch[key] = ch
        self.ok = True

    # ------------------------------------------------------------------ public API
    @property
    def effective_volume(self) -> float:
        return 0.0 if self.muted else self.volume

    def set_volume(self, v: float) -> None:
        self.volume = float(max(0.0, min(1.0, v)))
        self._apply_music(0.0)

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        self._apply_music(0.0)
        return self.muted

    def set_combat(self, on: bool) -> None:
        self._combat = bool(on)

    def on_ui(self, kind: str = "click") -> None:
        self.play("ui_click" if kind == "click" else "ui_hover", 1.0, 0.0)

    def play(self, name: str, vol: float = 1.0, pan: float = 0.0, delay: float = 0.0) -> None:
        if not self.ok:
            return
        try:
            if delay > 0:
                self._queue.append((self._clock + delay, name, vol, pan))
                if len(self._queue) > 64:
                    del self._queue[:16]
                return
            self._play_now(name, vol, pan)
        except Exception:  # noqa: BLE001
            log.debug("audio play failed", exc_info=True)

    def _play_now(self, name: str, vol: float, pan: float) -> None:
        snds = self._sounds.get(name)
        if not snds or self.effective_volume <= 0:
            return
        now = time.monotonic()
        if now - self._last_play.get(name, 0.0) < 0.045:
            return
        self._last_play[name] = now
        s = self._rng.choice(snds)
        ch = s.play()
        if ch is None:
            return
        g = BASE_GAIN.get(name, 0.5) * vol * self._rng.uniform(0.9, 1.08) * self.effective_volume
        pan = max(-1.0, min(1.0, pan))
        left = g * (1.0 - max(0.0, pan) * 0.7)
        right = g * (1.0 + min(0.0, pan) * 0.7)
        ch.set_volume(min(1.0, left), min(1.0, right))

    def update(self, dt: float) -> None:
        if not self.ok:
            return
        try:
            self._clock += dt
            if self._queue:
                due = [q for q in self._queue if q[0] <= self._clock]
                if due:
                    self._queue = [q for q in self._queue if q[0] > self._clock]
                    for _, name, vol, pan in due:
                        self._play_now(name, vol, pan)
            self._apply_music(dt)
        except Exception:  # noqa: BLE001
            log.debug("audio update failed", exc_info=True)

    def _apply_music(self, dt: float) -> None:
        if not self.ok:
            return
        k = 1.0 - np.exp(-dt / 1.2) if dt > 0 else 1.0
        tgt_c = 0.55 if self._combat else 0.0
        tgt_a = 0.5 - 0.3 * (1.0 if self._combat else 0.0)
        for key, tgt in (("ambient", tgt_a), ("combat", tgt_c)):
            self._music_vol[key] += (tgt - self._music_vol[key]) * k
            ch = self._music_ch.get(key)
            if ch is not None:
                ch.set_volume(float(self._music_vol[key] * self.effective_volume))

    def close(self) -> None:
        if not self.ok:
            return
        try:
            self._pg.mixer.stop()
            self._pg.mixer.quit()
        except Exception:  # noqa: BLE001
            pass
        self.ok = False
        self._sounds.clear()

    # ------------------------------------------------------------------ event mapping
    def on_event(self, d: dict, pan: float = 0.0) -> None:
        if not self.ok:
            return
        try:
            self._on_event(d, pan)
        except Exception:  # noqa: BLE001
            log.debug("audio on_event failed for %s", d, exc_info=True)

    def _on_event(self, d: dict, pan: float) -> None:  # noqa: C901
        t = d.get("t")
        data = d.get("data") or {}
        src, dst, amt, kind = d.get("src", -1), d.get("dst", -1), d.get("amount", 0) or 0, d.get("kind", -1)
        p = self.play
        now = time.monotonic()
        if t == ET.ATTACK:
            ranged = bool(data.get("ranged"))
            self._last_attack[src] = (now, ranged)
            p("bow_twang" if ranged else "sword_swing", 1.0, pan)
            if data.get("hit") and not ranged and (data.get("crit") or (data.get("ac") or 0) >= 16) and self._rng.random() < 0.6:
                p("clang", 0.8 if not data.get("crit") else 1.0, pan, 0.08)
            if data.get("crit"):
                p("sparkle", 0.35, pan, 0.12)
        elif t == ET.DAMAGE:
            k = int(kind) if kind is not None else 1
            big = min(1.3, 0.7 + amt / 20.0)
            last = self._last_attack.get(src)
            if k in _PHYS:
                if last and now - last[0] < 1.5 and last[1]:
                    p("arrow_impact", big, pan, 0.18)
                else:
                    p("flesh_hit", big, pan, 0.06)
            elif k == 3:
                p("explosion" if amt >= 18 else "fire_whoosh", big * 0.7, pan, 0.1)
            elif k == 2:
                p("ice_crackle", 0.9, pan, 0.05)
            elif k == 5:
                p("lightning_crack", 0.9, pan, 0.03)
            elif k == 12:
                p("thunder_boom", 0.7, pan)
            elif k == 6:
                p("necrotic_hiss", 0.9, pan)
            elif k in (0, 8):
                p("poison_bubble", 0.9, pan)
            elif k == 10:
                p("magic_chime", 0.7, pan)
            elif k == 9:
                p("sparkle", 0.7, pan)
            elif k == 4:
                p("flesh_hit", 0.7, pan)
                p("magic_chime", 0.4, pan, 0.05)
            if data.get("crit"):
                p("clang", 0.5, pan, 0.02)
        elif t == ET.SPELL_CAST:
            v = str(data.get("vfx") or data.get("spell") or "")
            if v in _FIRE:
                p("fire_whoosh", 1.0, pan)
            elif v in _ICE:
                p("ice_crackle", 0.9, pan)
            elif v in _LIGHTNING:
                p("lightning_crack", 0.8, pan)
            elif v in _THUNDER:
                p("thunder_boom", 0.8, pan)
            elif v in _HEAL or "heal" in v or "cure" in v:
                p("heal_shimmer", 1.0, pan)
            elif v in _NECRO:
                p("necrotic_hiss", 0.9, pan)
            elif v in _POISON:
                p("poison_bubble", 0.9, pan)
            else:
                p("magic_chime", 1.0, pan)
            p("sparkle", 0.5, pan, 0.1)
        elif t == ET.ABILITY_USE:
            v = str(data.get("vfx") or data.get("ability") or "")
            if v in _HEAL:
                p("heal_shimmer", 0.8, pan)
            elif v in ("rage", "action_surge", "smite", "stunning_strike", "sneak_attack"):
                p("clang", 0.5, pan)
                p("sparkle", 0.4, pan, 0.05)
            else:
                p("sparkle", 0.6, pan)
        elif t == ET.HEAL:
            p("heal_shimmer", 0.8, pan)
        elif t == ET.TEMP_HP:
            p("sparkle", 0.5, pan)
        elif t == ET.EFFECT_START:
            p("magic_chime", 0.4, pan)
        elif t == ET.CONDITION_ADD:
            p("poison_bubble" if kind == 9 else "magic_chime", 0.5, pan)
        elif t in (ET.SAVE, ET.CHECK, ET.CONCENTRATION_CHECK, ET.DEATH_SAVE):
            p("dice_rattle", 0.6, pan)
        elif t == ET.DEATH:
            p("death_thud", 1.0, pan, 0.1)
        elif t == ET.DOWNED:
            p("flesh_hit", 0.8, pan)
        elif t == ET.MOVE:
            steps = int(max(1, min(4, amt // 5 if amt else 1)))
            for i in range(steps):
                p("footstep_a" if i % 2 == 0 else "footstep_b", 0.9, pan, 0.02 + i * 0.17)
        elif t == ET.FORCED_MOVE:
            p("flesh_hit", 0.6, pan)
        elif t == ET.FALL:
            p("death_thud", 0.7, pan)
        elif t == ET.TELEPORT:
            p("sparkle", 0.9, pan)
        elif t == ET.DOOR:
            p("door_creak", 1.0, pan)
            if data.get("forced"):
                p("flesh_hit", 0.9, pan, 0.05)
        elif t == ET.CHEST_OPEN:
            p("chest_open", 1.0, pan)
        elif t == ET.LOCKPICK:
            p("ui_click", 0.8, pan)
            p("ui_click", 0.6, pan, 0.15)
        elif t in (ET.PICKUP, ET.GIVE, ET.GOLD_CHANGE):
            if amt:
                p("coin_jingle", 0.9, pan)
        elif t == ET.TRAP_TRIGGER:
            p("clang", 0.7, pan)
            p("flesh_hit", 0.6, pan, 0.1)
        elif t == ET.LEVER:
            p("clang", 0.4, pan)
        elif t == ET.COMBAT_START:
            self.set_combat(True)
            p("horn_swell", 1.0, 0.0)
        elif t == ET.COMBAT_END:
            self.set_combat(False)
            p("sparkle", 0.4, 0.0)
        elif t == ET.EPISODE_END:
            self.set_combat(False)
            winners = data.get("winners")
            p("horn_win" if winners else "horn_lose", 1.0, 0.0)
        elif t == ET.OBJECTIVE:
            p("magic_chime", 0.9, 0.0)
            p("heal_shimmer", 0.5, 0.0, 0.2)
        elif t == ET.BETRAYAL:
            p("horn_bad", 1.0, 0.0)
        elif t in (ET.TRUCE_ACCEPT, ET.TRUCE_OFFER):
            p("magic_chime", 0.6, 0.0)
        elif t == ET.SAY:
            p("ui_hover", 1.2, pan)
        elif t in (ET.ZONE_START, ET.SUMMON, ET.TRANSFORM):
            p("magic_chime", 0.7, pan)
        elif t == ET.COUNTERSPELL:
            p("lightning_crack", 0.6, pan)
            p("magic_chime", 0.6, pan, 0.05)
        elif t in (ET.REVIVED, ET.STABILIZED, ET.REST):
            p("heal_shimmer", 0.7, pan)
        elif t == ET.SURRENDER or t == ET.FLEE:
            p("ui_hover", 0.8, pan)
