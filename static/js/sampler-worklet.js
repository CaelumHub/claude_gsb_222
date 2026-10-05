/* sampler-worklet.js — Polyphonic sampler instrument (AudioWorklet).
 *
 * Mirrors backend/sampler.py DSP so an offline render of a recorded
 * performance sounds like the live play-through:
 *
 *   mode "varispeed" — linear-interpolated variable-rate playback; pitch and
 *                      speed/duration change together.
 *   mode "fixed"     — two Hann-grained crossfaded readers moving at rate r
 *                      while output advances 1:1: pitch shifts, duration
 *                      stays the same.
 *
 * Voices may be one-shot or loop between two loop points; the loop seam is
 * hidden with a linear crossfade.  Every voice has an ADSR-style envelope and
 * an optional one-pole low-pass ("timbre") filter.
 *
 * Protocol (messages from the main thread):
 *   {type:'load', buffer: Float32Array[], sr, loop:[ls,le], xfade:n, mode, loopOn}
 *   {type:'note', midi, rate, volume}        — start a voice
 *   {type:'off', midi}                       — release all voices on the note
 *   {type:'offAll'}                          — release every voice
 *   {type:'params', attack,decay,sustain,release,cutoff}
 */

const GRAIN = 0.06; // seconds, matches GRAIN_SEC in sampler.py

/* Round-half-to-even, matching Python's built-in round() (keeps frame-domain
 * constants — envelope times, grain sizes — identical to the offline renderer). */
function roundPy(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

class SamplerVoice {
  constructor(sample, sr, rate, volume, p, loop) {
    this.s = sample;           // array of Float32Array (per channel)
    this.ch = sample.length;
    this.end = sample[0].length;
    this.sr = sr;
    this.rate = rate;
    this.gain = volume;
    this.mode = p.mode;
    this.ls = loop.ls;
    this.lpLen = loop.lpLen;
    this.x = loop.x;
    this.le = loop.le;
    this.loopOn = loop.on && this.lpLen > 1;

    this.att = Math.max(1, roundPy(p.attack * sr));
    this.dec = Math.max(0, roundPy((p.decay || 0) * sr));
    this.sus = p.sustain;
    this.relN = Math.max(1, roundPy(p.release * sr));

    this.g = 0.0;               // envelope gain
    this.stage = 0;             // 0 A, 1 D, 2 S, 3 R, 4 done
    this.age = 0;
    this.t = 0.0;               // source cursor
    this.active = true;
    this.lpG = p.cutoff > 0 ? 1 - Math.exp(-2 * Math.PI * p.cutoff / sr) : 0;
    this.y = new Float64Array(this.ch);

    // Granular fixed-mode tables.
    this.grainN = Math.max(32, roundPy(GRAIN * sr));
    this.period = 2 * this.grainN;
    this.win = new Float32Array(2 * this.grainN);
    for (let i = 0; i < 2 * this.grainN; i++) {
      this.win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (2 * this.grainN - 1)));
    }
  }

  loopOne(k, c) {
    k = ((k % this.lpLen) + this.lpLen) % this.lpLen;
    let v = this.s[c][this.ls + k];
    if (this.x > 0 && k < this.x) {
      const a = k / this.x;
      v = v * a + this.s[c][this.le - this.x + k] * (1 - a);
    }
    return v;
  }

  readFrac(pos, c) {
    if (this.loopOn) {
      const k = ((pos - this.ls) % this.lpLen + this.lpLen) % this.lpLen;
      const i = k | 0, f = k - i;
      return this.loopOne(i, c) * (1 - f) + this.loopOne(i + 1, c) * f;
    }
    if (pos >= this.end - 1) return this.s[c][this.end - 1];
    const i = pos | 0, f = pos - i;
    return this.s[c][i] * (1 - f) + this.s[c][i + 1] * f;
  }

  readInt(idx, c) {
    if (this.loopOn) return this.loopOne(idx - this.ls, c);
    if (idx >= this.end || idx < 0) return 0;
    return this.s[c][idx];
  }

  release() { if (this.stage < 3) this.stage = 3; }

  process(out, n0, n1, master) {
    for (let j = n0; j < n1; j++) {
      let ended = false;
      for (let c = 0; c < this.ch; c++) {
        let v;
        if (this.mode === "fixed") {
          const i = this.t | 0;
          const g1 = i % this.period;
          const base1 = i - g1;
          const g2 = (i + this.grainN) % this.period;
          const base2 = i + this.grainN - g2;
          v = this.readInt(base1 + g1, c) * this.win[g1]
            + this.readInt(base2 + g2, c) * this.win[g2];
          if (!this.loopOn && base1 + g1 >= this.end && base2 + g2 >= this.end) ended = true;
        } else {
          if (!this.loopOn && this.t >= this.end - 1) ended = true;
          v = this.readFrac(this.t, c);
        }
        if (c === 0) {
          // Advance the envelope once per frame; natural sample end releases
          // one-shot voices even while the key is still held.
          if (ended) this.stage = 3;
          if (this.stage === 0) {
            this.age++;
            this.g = this.age / this.att;
            if (this.age >= this.att) this.stage = 1;
          } else if (this.stage === 1) {
            if (this.dec <= 0 || this.sus >= 1) {
              this.g = this.sus;
              this.stage = 2;
            } else {
              this.g -= (1 - this.sus) / this.dec;
              if (this.g <= this.sus) { this.g = this.sus; this.stage = 2; }
            }
          } else if (this.stage === 2) {
            this.g = this.sus;
          } else if (this.stage === 3) {
            this.g -= 1 / this.relN;
            if (this.g <= 0) { this.g = 0; this.stage = 4; }
          }
          if (!ended) this.t += this.rate;
        }
        if (this.stage === 4) { this.active = false; return; }
        let y = v * this.g * this.gain * master;
        if (this.lpG > 0) {
          this.y[c] += this.lpG * (y - this.y[c]);
          y = this.y[c];
        }
        out[c][j] += y;
      }
    }
  }
}

class SamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sample = null;
    this.bufferSr = sampleRate;
    this.voices = [];
    this.maxVoices = 64;
    this.params = {
      mode: "varispeed",
      attack: 0.005, decay: 0.0, sustain: 1.0, release: 0.15,
      cutoff: 0, master: 0.9,
    };
    this.loop = { on: false, ls: 0, le: 0, lpLen: 0, x: 0 };

    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "load") {
        // Transfer the per-channel buffers into the worklet.
        this.sample = m.buffer;
        this.bufferSr = m.sr;
        this.setLoop(m.loop, m.xfade, m.mode, m.loopOn);
      } else if (m.type === "loop") {
        this.setLoop(m.loop, m.xfade, this.params.mode, m.loopOn);
      } else if (m.type === "params") {
        Object.assign(this.params, m.params);
      } else if (m.type === "note") {
        this.noteOn(m);
      } else if (m.type === "off") {
        for (const v of this.voices) if (v.midi === m.midi) v.release();
      } else if (m.type === "offAll") {
        for (const v of this.voices) v.release();
      }
    };
  }

  setLoop(loopSec, xfadeSec, mode, on) {
    if (!this.sample) return;
    const n = this.sample[0].length;
    const sr = this.bufferSr;
    let ls = Math.max(0, Math.min(roundPy(loopSec[0] * sr), n));
    let le = Math.max(0, Math.min(roundPy(loopSec[1] * sr), n));
    if (le - ls < 8) { ls = 0; le = n; }
    let x = Math.max(0, roundPy(xfadeSec * sr));
    if (2 * x > le - ls) x = Math.max(0, ((le - ls) / 4) | 0);
    this.loop = { on: !!on, ls, le, lpLen: le - ls - x, x };
    this.params.mode = mode;
  }

  noteOn(m) {
    if (!this.sample) return;
    if (this.voices.length >= this.maxVoices) {
      // Steal the oldest releasing voice, else the oldest voice.
      let idx = this.voices.findIndex(v => v.stage >= 3);
      if (idx < 0) idx = 0;
      this.voices.splice(idx, 1);
    }
    const v = new SamplerVoice(this.sample, this.bufferSr, m.rate, m.volume,
      { ...this.params }, { ...this.loop });
    v.midi = m.midi;
    this.voices.push(v);
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const n = out[0].length;
    const ch = this.sample ? this.sample.length : 1;
    // Render into an internal buffer so mismatched channel counts are handled.
    if (!this._buf || this._buf.length !== ch || this._buf[0].length !== n) {
      this._buf = Array.from({ length: ch }, () => new Float64Array(n));
    }
    const buf = this._buf;
    for (let c = 0; c < ch; c++) buf[c].fill(0);

    if (this.sample) {
      for (let i = this.voices.length - 1; i >= 0; i--) {
        const v = this.voices[i];
        v.process(buf, 0, n, this.params.master);
        if (!v.active) this.voices.splice(i, 1);
      }
    }

    // Map to output (mono sample -> both output channels if stereo).
    const outCh = out.length;
    for (let c = 0; c < outCh; c++) {
      const src = ch === 1 ? buf[0] : buf[Math.min(c, ch - 1)];
      const dst = out[c];
      for (let j = 0; j < n; j++) {
        let x = src[j];
        if (x > 1) x = 1; else if (x < -1) x = -1;
        dst[j] = x;
      }
    }
    return true;
  }
}

registerProcessor("sampler-processor", SamplerProcessor);
