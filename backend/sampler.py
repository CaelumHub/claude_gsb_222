"""
sampler.py — Sampler instrument: sample extraction + offline performance render.

A *sampler* loads one excerpt of an audio-library file and plays it back at
different speeds so that each key of a keyboard sounds a different pitch:

* **varispeed mode** — classic tape-style playback.  Pitch and playback speed
  change together (``rate = 2 ** (semitones / 12)``), so duration changes
  proportionally with pitch.
* **fixed mode** — a two-grain crossfaded granular pitch shifter.  Pitch
  changes while playback speed (and therefore duration) stays constant.

Samples can be **one-shot** (play once to the end) or **looping** between two
loop points; the loop seam is hidden by a crossfade of configurable length.

Every voice goes through an ADSR-style envelope and an optional one-pole
low-pass "tone" filter, and each note carries its own volume (the per-key
velocity set up in the UI).

The real-time instrument runs in an AudioWorklet in the browser (same DSP);
this module renders a recorded performance (a list of note events) offline,
streaming the mix straight into a WAV writer so even long performances with
wide ranges and dense, fast key presses mix with bounded memory.  Pure
standard-library Python, matching the rest of the backend.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Optional, Sequence

from . import audio_io

# --------------------------------------------------------------------------- #
# Limits & shared constants
# --------------------------------------------------------------------------- #

GRAIN_SEC = 0.06          # granular pitch-shifter grain length (fixed mode)
MAX_VOICES = 64           # polyphony cap; the quietest voices are stolen
MAX_REGION_SEC = 180.0    # guard for sample extraction
MAX_PERF_SEC = 600.0      # guard for rendered performances


def note_rate(semitones: float) -> float:
    """Playback rate for ``semitones`` away from the root note."""
    return 2.0 ** (semitones / 12.0)


# --------------------------------------------------------------------------- #
# Sample extraction
# --------------------------------------------------------------------------- #

def extract_sample(path: str, start: float, end: Optional[float] = None
                   ) -> audio_io.AudioData:
    """Load the [start, end) seconds region of a library file into memory.

    Library files are always PCM WAV (uploads get converted), but non-WAV
    sources are decoded with ffmpeg as a fallback.
    """
    if not path.lower().endswith(".wav") or not os.path.isfile(path):
        wav_path = audio_io.decode_with_ffmpeg(path)
        tmp = True
    else:
        wav_path = path
        tmp = False
    try:
        with audio_io.WavReader(wav_path) as r:
            sr = r.sr
            total = r.nframes
            a = max(0, min(int(round(start * sr)), total))
            if end is None:
                b = total
            else:
                b = max(a, min(int(round(end * sr)), total))
            excerpt = r.read_excerpt(a, b - a)
        if excerpt.duration > MAX_REGION_SEC:
            raise ValueError(f"采样区域过长（上限 {MAX_REGION_SEC:.0f} 秒）")
        return excerpt
    finally:
        if tmp and os.path.exists(wav_path):
            os.unlink(wav_path)


# --------------------------------------------------------------------------- #
# Configuration normalisation
# --------------------------------------------------------------------------- #

def _norm_config(cfg: Dict, sr: int, length: int) -> Dict:
    """Turn the JSON config (seconds relative to the extracted region) into
    frame-domain values.  ``length`` is the region length in frames."""
    out = {
        "root_midi": int(cfg.get("root_midi", 60)),
        "mode": cfg.get("mode", "varispeed"),
        "loop": bool(cfg.get("loop", False)),
        "attack": max(0.0, float(cfg.get("attack", 0.005))),
        "decay": max(0.0, float(cfg.get("decay", 0.0))),
        "sustain": min(1.0, max(0.0, float(cfg.get("sustain", 1.0)))),
        "release": max(0.005, float(cfg.get("release", 0.15))),
        "cutoff": float(cfg.get("cutoff", 0.0)),
        "master": min(2.0, max(0.0, float(cfg.get("master", 0.9)))),
        "xfade": max(0.0, float(cfg.get("loop_xfade", 0.02))),
    }
    if out["mode"] not in ("varispeed", "fixed"):
        out["mode"] = "varispeed"

    # Loop points are seconds relative to the extracted region start.
    ls = max(0, min(int(round(float(cfg.get("loop_start", 0.0)) * sr)), length))
    le = max(0, min(int(round(float(cfg.get("loop_end", 0.0)) * sr)), length))
    if le - ls < 8:
        ls, le = 0, length
    x = int(round(out["xfade"] * sr))
    if 2 * x > le - ls:
        x = max(0, (le - ls) // 4)
    out.update({"ls": ls, "le": le, "lp_len": le - ls - x, "x": x})
    return out


# --------------------------------------------------------------------------- #
# Single playing voice (streaming — one block at a time)
# --------------------------------------------------------------------------- #

class Voice:
    """One triggered note.

    Time is driven by :meth:`render_block`, which writes the voice's
    contribution for an arbitrary global block.  Source position, envelope
    stage and filter state are all carried between calls so a voice spans as
    many blocks as it needs without buffering the whole note.
    """

    def __init__(self, data: List[List[float]], sr: int, ev: Dict, cfg: Dict):
        self.data = data
        self.sr = sr
        self.channels = len(data)
        self.end = len(data[0]) if data else 0
        self.start = max(0, int(round(float(ev["start"]) * sr)))
        dur = max(0.02, float(ev.get("duration", 1.0)))
        self.hold_n = int(round(dur * sr))  # held frames after note-on
        self.gain = min(2.0, max(0.0, float(ev.get("volume", 1.0))))
        self.rate = note_rate(float(ev["midi"]) - cfg["root_midi"])

        self.mode = cfg["mode"]
        self.loop = cfg["loop"]
        self.ls, self.le, self.lp_len, self.x = cfg["ls"], cfg["le"], cfg["lp_len"], cfg["x"]

        self.att = max(1, int(round(cfg["attack"] * sr)))
        self.dec = max(0, int(round(cfg["decay"] * sr)))
        self.sus = cfg["sustain"]
        self.rel_n = max(1, int(round(cfg["release"] * sr)))

        # One-pole low-pass state per channel.
        self.cutoff = cfg["cutoff"]
        self.lp_g = (1.0 - math.exp(-2.0 * math.pi * self.cutoff / sr)) if self.cutoff > 0 else 0.0
        self.lp_y = [0.0] * self.channels

        # Granular fixed-mode state.
        self.grain = max(32, int(round(GRAIN_SEC * sr)))
        self.period = 2 * self.grain
        self.win = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (2 * self.grain - 1)))
                    for i in range(2 * self.grain)]

        # Playback cursors.
        self.t = 0.0          # source position in frames (data-relative)
        self.env = 0.0        # current envelope gain
        self.phase = 0        # 0 attack, 1 decay, 2 sustain, 3 release, 4 done
        self.age = 0          # frames since note-on
        self.active = True

    # -- source readers ---------------------------------------------------- #

    def _loop_one(self, k: int, c: int) -> float:
        """One crossfaded loop sample at integer offset ``k`` in [0, lp_len)."""
        k %= self.lp_len
        v = self.data[c][self.ls + k]
        if self.x > 0 and k < self.x:
            a = k / self.x
            v = v * a + self.data[c][self.le - self.x + k] * (1.0 - a)
        return v

    def _read_frac(self, pos: float, c: int) -> float:
        """Linear-interpolated read (used by varispeed mode)."""
        if self.loop and self.lp_len > 1:
            k = (pos - self.ls) % self.lp_len
            i = int(k)
            f = k - i
            a = self._loop_one(i, c)
            b = self._loop_one(i + 1, c)
            return a * (1.0 - f) + b * f
        i = int(pos)
        if i >= self.end - 1:
            return self.data[c][self.end - 1] if self.end else 0.0
        f = pos - i
        return self.data[c][i] * (1.0 - f) + self.data[c][i + 1] * f

    def _read_int(self, idx: int, c: int) -> float:
        """Integer read with loop crossfade (used by fixed mode)."""
        if idx < 0:
            idx = 0
        if self.loop and self.lp_len > 1:
            return self._loop_one(idx - self.ls, c)
        if idx >= self.end:
            return 0.0  # signals "past the end" (silence, not a clamped sample)
        return self.data[c][idx]

    # -- envelope ---------------------------------------------------------- #

    def _advance_env(self, held: bool) -> None:
        if self.phase == 0:  # attack
            self.age += 1
            self.env = self.age / self.att
            if self.age >= self.att:
                self.phase = 1
        elif self.phase == 1:  # decay
            if self.dec <= 0 or self.sus >= 1.0:
                self.env = self.sus
                self.phase = 2
            else:
                self.env -= (1.0 - self.sus) / self.dec
                if self.env <= self.sus:
                    self.env = self.sus
                    self.phase = 2
        elif self.phase == 2:  # sustain
            self.env = self.sus
        elif self.phase == 3:  # release
            self.env -= 1.0 / self.rel_n
        if not held and self.phase < 3:
            self.phase = 3
        if self.env <= 0.0 and self.phase >= 3:
            self.env = 0.0
            self.phase = 4
            self.active = False

    # -- block rendering --------------------------------------------------- #

    def render_block(self, gpos: int, n: int, out: List[List[float]],
                     master: float) -> None:
        if not self.active:
            return
        end_g = gpos + n
        if end_g <= self.start:
            return
        i0 = max(0, self.start - gpos)

        for j in range(i0, n):
            gf = gpos + j
            held = (gf - self.start) < self.hold_n

            # --- source value per channel at the current source position -- #
            vals = [0.0] * self.channels
            natural_end = False
            if self.mode == "fixed":
                i = int(self.t)
                g1 = i % self.period
                base1 = i - g1
                g2 = (i + self.grain) % self.period
                base2 = i + self.grain - g2
                w1, w2 = self.win[g1], self.win[g2]
                for c in range(self.channels):
                    vals[c] = self._read_int(base1 + g1, c) * w1 \
                        + self._read_int(base2 + g2, c) * w2
                if not self.loop and (base1 + g1 >= self.end
                                      and base2 + g2 >= self.end):
                    natural_end = True
                self.t += self.rate
            else:  # varispeed
                # One-shot: once the source is exhausted the voice finishes;
                # looping voices wrap inside the reader and never naturally end.
                if not self.loop and self.t >= self.end - 1:
                    natural_end = True
                for c in range(self.channels):
                    vals[c] = self._read_frac(self.t, c)
                if not natural_end:
                    self.t += self.rate

            self._advance_env(held and not natural_end)
            if not self.active:
                break

            g = self.env * self.gain * master
            for c in range(self.channels):
                y = vals[c] * g
                if self.lp_g > 0.0:  # one-pole low pass ("timbre")
                    self.lp_y[c] += self.lp_g * (y - self.lp_y[c])
                    y = self.lp_y[c]
                out[c][j] += y


# --------------------------------------------------------------------------- #
# Performance rendering
# --------------------------------------------------------------------------- #

def render_performance(sample: audio_io.AudioData, events: Sequence[Dict],
                       cfg: Dict, dst_path: str) -> Dict:
    """Render ``events`` (recorded note triggers) into ``dst_path`` (WAV).

    Each event is ``{midi, start, duration, volume}`` with times in seconds,
    measured from performance start.  The mix streams block-by-block into the
    WAV writer, so memory use is bounded regardless of performance length.
    """
    sr = sample.sr
    data = sample.samples
    if not data or not events:
        raise ValueError("没有可渲染的演奏事件")
    channels = sample.channels

    region_b = len(data[0])
    c = _norm_config(cfg, sr, region_b)

    # Build one voice per note-on/note-off pair.
    voices: List[Voice] = []
    last_end = 0.0
    for ev in events:
        if len(voices) >= MAX_VOICES:
            # Steal the oldest voice (dense passages keep the most recent
            # notes), matching the worklet's voice cap.
            voices.pop(0)
        try:
            v = Voice(data, sr, ev, c)
        except (KeyError, ValueError):
            continue
        voices.append(v)
        # A voice ends at key-off + release, or earlier when a one-shot sample
        # reaches its natural end (fixed: source length; varispeed: length /
        # playback rate).
        keyoff_end = float(ev["start"]) + float(ev.get("duration", 1.0))
        if not c["loop"]:
            if c["mode"] == "fixed":
                natural = float(ev["start"]) + (region_b - 1) / sr
            else:
                natural = float(ev["start"]) + (region_b - 1) / (sr * v.rate)
            voice_end = min(keyoff_end, natural)
        else:
            voice_end = keyoff_end
        last_end = max(last_end, voice_end + c["release"] + 0.02)
    total = min(int(round(last_end * sr)) + sr // 20,
                int(MAX_PERF_SEC * sr))
    total = max(total, 1)

    block = 1 << 13
    with audio_io.WavWriter(dst_path, sr, channels, 2) as w:
        gpos = 0
        while gpos < total:
            n = min(block, total - gpos)
            buf = [[0.0] * n for _ in range(channels)]
            alive = []
            for v in voices:
                v.render_block(gpos, n, buf, c["master"])
                if v.active:
                    alive.append(v)
            voices = alive
            # Hard clip guard against clipping from overlapping notes.
            for ch_idx in range(channels):
                row = buf[ch_idx]
                for k in range(n):
                    x_ = row[k]
                    if x_ > 1.0:
                        row[k] = 1.0
                    elif x_ < -1.0:
                        row[k] = -1.0
            w.write_chunk(buf)
            gpos += n

    return {
        "sr": sr,
        "channels": channels,
        "frames": total,
        "duration": total / sr,
        "notes": len(events),
    }


def render_to_temp(sample: audio_io.AudioData, events: Sequence[Dict],
                   cfg: Dict) -> str:
    """Convenience wrapper: render into a temp WAV and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    render_performance(sample, events, cfg, path)
    return path
