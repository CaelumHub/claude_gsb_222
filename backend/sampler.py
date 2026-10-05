"""
sampler.py — Sampler synthesis backend (pure Python, stdlib only).

A *sampler* takes one excerpt ("region") of a source recording and replays it
at different pitches so a single sound becomes a playable instrument.

Two playback models are supported, matching the front-end:

* ``tape``  — varispeed.  Pitch is changed by resampling, exactly like tape
  speed: raising the pitch shortens the note and vice versa.  Implemented with
  linear-interpolation resampling (see :mod:`backend.dsp`).
* ``pitch`` — granular overlap-add pitch shifting.  Short Hann-windowed grains
  are replayed at the requested rate but their read head advances 1:1 with the
  output clock (``hop = G/2``), so the **pitch changes while the duration stays
  constant**.  Hann windows at 50 % overlap satisfy COLA (their sum is 1), so
  overlapping grains reconstruct the signal without amplitude modulation.

Triggering:

* ``oneshot`` — the region plays through to the end (note-off is ignored).
* ``loop``    — the region plays as an intro, then the loop segment repeats
  until note-off, with an attack/release envelope avoiding clicks.

The public entry point is :func:`render_performance`, which mixes a list of
note events (MIDI key, start time, duration, velocity, per-key gain) into one
WAV file.  Every note buffer for a given pitch ratio is cached and reused, so
dense passages with repeated notes stay cheap.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp

# --------------------------------------------------------------------------- #
# Constraints — offline renders are bounded so a malformed client cannot
# request an unbounded amount of pure-Python number crunching.
# --------------------------------------------------------------------------- #

MAX_EVENTS = 512
MAX_SECONDS = 600.0
DEFAULT_GRAIN_MS = 0.12
MAX_REGION_SECONDS = 60.0


# --------------------------------------------------------------------------- #
# Region preparation
# --------------------------------------------------------------------------- #

def load_region(path: str, start: float, end: float,
                max_seconds: float = MAX_REGION_SECONDS) -> Tuple[List[float], int]:
    """Extract ``[start, end]`` seconds of a file as a mono float list.

    Channels are averaged; the source sample rate is preserved (the sampler
    renders at the source rate).
    """
    with audio_io.WavReader(path) as r:
        sr = r.sr
        total = r.nframes
        a = max(0, int(round(max(0.0, start) * sr)))
        b = min(total, int(round(end * sr)))
        if b <= a:
            raise ValueError("采样区间为空（终点必须晚于起点）")
        if b - a > int(max_seconds * sr):
            raise ValueError(f"采样区间过长（上限 {max_seconds:g}s）")
        r._w.setpos(a)
        chunk = r.read_chunk(b - a)
    if not chunk:
        raise ValueError("无法读取采样区间")
    mono = audio_io.to_mono(chunk)
    return mono[: b - a], sr


# --------------------------------------------------------------------------- #
# Small DSP helpers
# --------------------------------------------------------------------------- #

def pitch_ratio(midi: float, root: float) -> float:
    return 2.0 ** ((midi - root) / 12.0)


def _interp(s: Sequence[float], i: float, n: int) -> float:
    """Linear interpolation with edge clamp."""
    if i <= 0.0:
        return s[0]
    i0 = int(i)
    if i0 >= n - 1:
        return s[n - 1]
    f = i - i0
    return s[i0] * (1.0 - f) + s[i0 + 1] * f


def _read_loop(s: Sequence[float], i: float, n: int, loop0: int, loop1: int) -> float:
    """Read at fractional position ``i``; once past ``loop1`` wrap the loop."""
    if i < loop1:
        return _interp(s, i, n)
    span = loop1 - loop0
    if span <= 0:
        return _interp(s, i, n)
    return _interp(s, loop0 + (i - loop0) % span, n)


# --------------------------------------------------------------------------- #
# Note rendering
# --------------------------------------------------------------------------- #

def _grain_shift(samples, ratio, n_out, sr, grain_ms, read):
    """Shared granular pitch-shift kernel.

    "Stretch by ``ratio`` then resample by 1/ratio" in one pass: Hann grains
    (length ``G``) are *placed* every ``G/2`` output frames, so the output
    keeps the source length; within a grain, output offset ``j`` reads the
    source at offset ``ratio*j`` around the grain centre (which advances at
    the same ``G/2`` rate), so the pitch is multiplied by ``ratio``.  The
    window is evaluated at the *source* phase and the overlap weights are
    normalised per output frame, keeping the reconstruction flat for any
    ratio (including ratio < 1 where grains overlap more than two-deep).

    ``read(samples, idx, n)`` supplies source samples — plain for one-shots
    and wrapping for loops.
    """
    g = max(64, int(round(grain_ms * sr)))
    hop = g // 2
    half = g / 2.0
    two_pi = 2.0 * math.pi
    out = [0.0] * n_out
    wsum = [0.0] * n_out
    # Only |j| up to reach can fall inside the window's [0, G) source span.
    reach = int(math.ceil(half / ratio))
    center = hop
    k = 1
    while center - reach < n_out:
        oc = int(round(center))
        j_lo = max(-reach, -oc)
        j_hi = min(reach, n_out - 1 - oc)
        for j in range(j_lo, j_hi + 1):
            f = ratio * j
            ph = (f + half) / g          # window phase in (0, 1)
            if ph <= 0.0 or ph >= 1.0:
                continue
            w = 0.5 - 0.5 * math.cos(two_pi * ph)
            oi = oc + j
            out[oi] += w * read(samples, center + f, len(samples))
            wsum[oi] += w
        k += 1
        center = k * hop
    for i in range(n_out):
        w = wsum[i]
        if w > 1e-6:
            out[i] /= w
    return out


def _pitch_shift_oneshot(samples: Sequence[float], ratio: float, sr: int,
                         grain_ms: float = DEFAULT_GRAIN_MS) -> List[float]:
    """Granular pitch shift; output has the same length as the input."""
    if not samples:
        return []
    if abs(ratio - 1.0) < 1e-6:
        return list(samples)
    return _grain_shift(samples, ratio, len(samples), sr, grain_ms, _interp)


def _pitch_shift_loop(samples: Sequence[float], ratio: float, dur: int,
                      loop0: int, loop1: int, sr: int,
                      grain_ms: float = DEFAULT_GRAIN_MS) -> List[float]:
    """Granular pitch shift of a sustaining, looping note of ``dur`` frames.

    Grain centres walk the intro once and then circle the loop segment; on
    the output they keep advancing at ``G/2`` per grain.
    """
    if dur <= 0:
        return []
    n = len(samples)
    g = max(64, int(round(grain_ms * sr)))
    hop = g // 2
    half = g / 2.0
    two_pi = 2.0 * math.pi
    out = [0.0] * dur
    wsum = [0.0] * dur
    reach = int(math.ceil(half / ratio))
    span = max(1, loop1 - loop0)
    center = hop
    while center - reach < dur:
        # Map the output-centre timeline onto intro-then-loop source positions.
        sc = center if center < loop1 else loop0 + (center - loop0) % span
        oc = int(round(center))
        j_lo = max(-reach, -oc)
        j_hi = min(reach, dur - 1 - oc)
        for j in range(j_lo, j_hi + 1):
            f = ratio * j
            ph = (f + half) / g
            if ph <= 0.0 or ph >= 1.0:
                continue
            w = 0.5 - 0.5 * math.cos(two_pi * ph)
            oi = oc + j
            out[oi] += w * _read_loop(samples, sc + f, n, loop0, loop1)
            wsum[oi] += w
        center += hop
    for i in range(dur):
        w = wsum[i]
        if w > 1e-6:
            out[i] /= w
    return out


def _tape_oneshot(samples: Sequence[float], ratio: float) -> List[float]:
    """Varispeed replay: y[i] = x[i*ratio], length n/ratio (duration scales)."""
    n = len(samples)
    m = int(round(n / ratio))
    out = [0.0] * m
    pos = 0.0
    for i in range(m):
        out[i] = _interp(samples, pos, n)
        pos += ratio
    return out


def _tape_loop(samples: Sequence[float], ratio: float, dur: int,
               loop0: int, loop1: int) -> List[float]:
    """Varispeed replay of a sustaining, looping note of ``dur`` frames."""
    n = len(samples)
    out = [0.0] * dur
    pos = 0.0
    for i in range(dur):
        out[i] = _read_loop(samples, pos, n, loop0, loop1)
        pos += ratio
    return out


def _apply_env(samples: List[float], sr: int, attack: float, release: float,
               release_from: Optional[int] = None) -> None:
    """In-place attack (always) and release envelope to avoid clicks."""
    n = len(samples)
    if n == 0:
        return
    atk = max(1, int(attack * sr))
    for i in range(min(atk, n)):
        samples[i] *= i / atk
    if release > 0 and release_from is not None and release_from < n:
        rel = max(1, int(release * sr))
        for j in range(n - release_from):
            if j >= rel:
                samples[release_from + j] = 0.0
            else:
                samples[release_from + j] *= 1.0 - j / rel


def render_note(region: Sequence[float], sr: int, ratio: float, mode: str,
                trigger: str, note_frames: Optional[int],
                loop0: int, loop1: int,
                attack: float = 0.005, release: float = 0.09) -> List[float]:
    """Render one note (without the event gain; gain is applied at mix time).

    ``note_frames`` is the held length for looping notes (ignored for one-shots,
    whose length is intrinsic).
    """
    n = len(region)
    if trigger == "loop":
        if loop1 <= 0 or loop0 >= loop1:
            raise ValueError("循环区间无效（需要在采样区间内部）")
        if note_frames is None or note_frames <= 0:
            raise ValueError("循环音符缺少时长")
        rel_frames = int(release * sr)
        dur = note_frames + rel_frames
        if mode == "tape":
            out = _tape_loop(region, ratio, dur, loop0, loop1)
        else:
            out = _pitch_shift_loop(region, ratio, dur, loop0, loop1, sr)
        _apply_env(out, sr, attack, release, release_from=note_frames)
        return out

    # One-shot: plays through to the end, note-off is ignored.
    if mode == "tape":
        out = _tape_oneshot(region, ratio)
    else:
        out = _pitch_shift_oneshot(region, ratio, sr)
    _apply_env(out, sr, attack, 0.0)
    return out


# --------------------------------------------------------------------------- #
# Performance / timeline mix
# --------------------------------------------------------------------------- #

def _normalise_config(cfg: Dict) -> Dict:
    mode = cfg.get("mode", "pitch")
    if mode not in ("tape", "pitch"):
        raise ValueError("mode 必须是 tape 或 pitch")
    trigger = cfg.get("trigger", "oneshot")
    if trigger not in ("oneshot", "loop"):
        raise ValueError("trigger 必须是 oneshot 或 loop")
    try:
        root = float(cfg.get("root", 60))
        start = float(cfg.get("region_start", 0.0))
        end = float(cfg.get("region_end", 0.0))
    except (TypeError, ValueError):
        raise ValueError("root / region 参数必须是数字")
    if end <= start:
        raise ValueError("采样区间终点必须晚于起点")
    loop_start = float(cfg.get("loop_start", start))
    loop_end = float(cfg.get("loop_end", end))
    if trigger == "loop" and not (start <= loop_start < loop_end <= end):
        raise ValueError("循环区间必须位于采样区间内部")
    return {
        "mode": mode,
        "trigger": trigger,
        "root": root,
        "region_start": start,
        "region_end": end,
        "loop_start": loop_start,
        "loop_end": loop_end,
    }


def render_performance(src_path: str, config: Dict, events: List[Dict],
                       dst_path: Optional[str] = None,
                       attack: float = 0.005, release: float = 0.09
                       ) -> Tuple[str, int, int, float]:
    """Render a recorded performance to a mono 16-bit WAV file.

    Each event is ``{midi, start (seconds), duration (seconds or null),
    velocity, gain}``.  One-shot events derive their length from the sample;
    loop events use ``duration``.  Returns ``(path, sr, frames, seconds)``.
    """
    cfg = _normalise_config(config or {})
    if not isinstance(events, list) or not events:
        raise ValueError("没有可渲染的音符事件")
    if len(events) > MAX_EVENTS:
        raise ValueError(f"音符事件过多（上限 {MAX_EVENTS}）")

    region, sr = load_region(src_path, cfg["region_start"], cfg["region_end"])
    n = len(region)
    loop0 = max(0, int(round((cfg["loop_start"] - cfg["region_start"]) * sr)))
    loop1 = min(n, int(round((cfg["loop_end"] - cfg["region_start"]) * sr)))

    # Cache rendered note buffers keyed by everything that affects their shape;
    # the (per-performance) event gain is applied at mix time instead.
    cache: Dict[Tuple, List[float]] = {}

    placements: List[Tuple[int, List[float], float]] = []
    total = 0
    for ev in events:
        try:
            midi = float(ev["midi"])
            t0 = max(0.0, float(ev.get("start", 0.0)))
            vel = max(0.0, min(1.0, float(ev.get("velocity", 1.0))))
            gain = max(0.0, min(4.0, float(ev.get("gain", 1.0))))
        except (KeyError, TypeError, ValueError):
            raise ValueError("音符事件格式不正确")
        dur = ev.get("duration")
        if dur is not None:
            dur = max(0.0, float(dur))
            if dur > MAX_SECONDS:
                raise ValueError(f"音符时长超出上限（{MAX_SECONDS:g}s）")
        note_frames = None if dur is None else int(round(dur * sr))
        ratio = pitch_ratio(midi, cfg["root"])
        key = (round(ratio, 6), note_frames)
        buf = cache.get(key)
        if buf is None:
            buf = render_note(region, sr, ratio, cfg["mode"], cfg["trigger"],
                              note_frames, loop0, loop1, attack=attack, release=release)
            cache[key] = buf
        pos = int(round(t0 * sr))
        placements.append((pos, buf, gain * vel))
        total = max(total, pos + len(buf))

    if total > int(MAX_SECONDS * sr):
        raise ValueError(f"渲染结果过长（上限 {MAX_SECONDS:g}s）")
    if total <= 0:
        raise ValueError("渲染结果为空")

    mix = [0.0] * total
    for pos, buf, g in placements:
        if g <= 0.0:
            continue
        end = min(total, pos + len(buf))
        for i in range(pos, end):
            mix[i] += buf[i - pos] * g
    # Soft clip so dense chords cannot hard-clip.
    for i, v in enumerate(mix):
        if v > 1.0 or v < -1.0:
            mix[i] = math.tanh(v)

    if dst_path is None:
        fd, dst_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    audio_io.save(dst_path, audio_io.AudioData([mix], sr))
    return dst_path, sr, total, total / sr
