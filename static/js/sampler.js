/* sampler.js — 采样器：采样音色 + 实时演奏 + 录音
 *
 * Two synthesis paths mirror backend/sampler.py:
 *
 *   tape  — varispeed via native AudioBufferSourceNode.playbackRate.
 *           Pitch changes AND duration/speed changes proportionally (like tape).
 *   pitch — self-written granular overlap-add pitch shifter running in a
 *           ScriptProcessorNode.  Grains are Hann-windowed, placed every G/2
 *           output frames and read the source at `ratio*j` around each grain
 *           centre with per-frame overlap-weight normalisation; pitch changes
 *           while duration stays constant.
 *
 * Triggering: oneshot (plays through, note-off ignored) or loop
 * (intro + loop segment repeats until note-off, enveloped to avoid clicks).
 */

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const midiToName = (m) => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

const SCALES = {
  chromatic: { label: "半音阶", offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] },
  major:     { label: "自然大调", offsets: [0, 2, 4, 5, 7, 9, 11] },
  minor:     { label: "自然小调", offsets: [0, 2, 3, 5, 7, 8, 10] },
  penta:     { label: "五声音阶", offsets: [0, 2, 4, 7, 9] },
  pentamin:  { label: "小调五声", offsets: [0, 3, 5, 7, 10] },
  blues:     { label: "布鲁斯", offsets: [0, 3, 5, 6, 7, 10] },
};

// Computer keyboard: bottom row = white-style steps, top row = their sharps.
const KEY_ROWS = [
  ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"],
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
];
const KEY_ROW_SHARP = { z: "s", x: "d", c: "f", v: "g", b: "h", n: "j", m: "k", ",": "l", ".": ";", "/": undefined,
                        a: "w", s: "e", d: "r", f: "t", g: "y", h: "u", j: "i", k: "o", l: "p", ";": undefined };

/* ========================================================================= */
/* SamplerEngine                                                             */
/* ========================================================================= */

class SamplerEngine {
  constructor() {
    this.ctx = null;
    this.master = null;           // master gain -> compressor -> destination
    this.buffer = null;           // source AudioBuffer
    this.region = null;           // Float32Array view/copy of the selected region
    this.cfg = {};                // mode, trigger, root, loop frames (region-local)
    this.keyGains = {};           // midi -> 0..2
    this.velocity = 0.9;
    this.maxVoices = 32;
    this.tapeVoices = [];          // {midi, gain, src, gainNode}
    this.grainVoices = [];         // pitch-mode voices (mixed in one processor)
    this.grainNode = null;
    this.procSr = 44100;
    this.grainMs = 0.12;
    this.onVoicesChange = null;
    this.activeMidis = new Set();
  }

  async init() {
    if (this.ctx) { if (this.ctx.state === "suspended") await this.ctx.resume(); return; }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx();
    this.procSr = this.ctx.sampleRate;
    const comp = this.ctx.createDynamicsCompressor();
    comp.threshold.value = -10;
    comp.knee.value = 6;
    comp.ratio.value = 12;
    comp.attack.value = 0.003;
    comp.release.value = 0.12;
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.9;
    this.master.connect(comp);
    comp.connect(this.ctx.destination);
    this._startGrainProcessor();
  }

  /* The granular engine shares one ScriptProcessorNode; every render quantum
     it sums all active grain voices.  Kept alive for the page lifetime. */
  _startGrainProcessor() {
    const BUF = 2048;
    const node = this.ctx.createScriptProcessor(BUF, 1, 1);
    node.onaudioprocess = (e) => this._renderGrains(e.outputBuffer.getChannelData(0));
    // ScriptProcessor only runs while connected to destination.
    node.connect(this.master);
    this.grainNode = node;
  }

  async loadFile(fileId, regionStart, regionEnd) {
    await this.init();
    // Silence anything still ringing from the previous sample.
    this.activeMidis.clear();
    this.tapeVoices.slice().forEach((v) => { if (!v.stopAt) { try { v.src.stop(); } catch (_) {} } });
    this.grainVoices.length = 0;
    const res = await fetch(API.fileUrl(fileId));
    const arr = await res.arrayBuffer();
    this.buffer = await this.ctx.decodeAudioData(arr);
    this.setRegion(regionStart, regionEnd);
  }

  setRegion(startS, endS) {
    if (!this.buffer) return;
    this.regionStartAbs = startS;
    const sr = this.buffer.sampleRate;
    let a = Math.max(0, Math.round(startS * sr));
    let b = Math.min(this.buffer.length, Math.round(endS * sr));
    if (b <= a) b = Math.min(this.buffer.length, a + 1);
    // Downmix region channels into one local Float32Array.
    const ch = this.buffer.numberOfChannels;
    if (ch === 1) {
      this.region = this.buffer.getChannelData(0).subarray(a, b);
    } else {
      const reg = new Float32Array(b - a);
      for (let c = 0; c < ch; c++) {
        const d = this.buffer.getChannelData(c);
        for (let i = 0; i < reg.length; i++) reg[i] += d[a + i] / ch;
      }
      this.region = reg;
    }
  }

  configure(cfg) {
    // cfg: {mode, trigger, root, loopStart, loopEnd} seconds (region-relative)
    this.cfg = cfg;
    this.cfg.regionStartAbs = this.regionStartAbs || 0;
    const proc = this.procSr;
    const srcSr = this.buffer ? this.buffer.sampleRate : proc;
    const lenProc = this.region ? this.region.length / srcSr * proc : 0;
    this.cfg.loop0 = Math.max(0, Math.round(cfg.loopStart * proc));
    this.cfg.loop1 = Math.min(lenProc, Math.round(cfg.loopEnd * proc));
  }

  now() { return this.ctx ? this.ctx.currentTime : 0; }

  /* -------------------------------------------------------------- note on */

  noteOn(midi, velocity) {
    if (!this.region) return;
    this.activeMidis.add(midi);
    const v = velocity || this.velocity;
    if (this.cfg.mode === "tape") this._tapeOn(midi, v);
    else this._grainOn(midi, v);
    this.onVoicesChange && this.onVoicesChange(this.activeMidis);
  }

  noteOff(midi) {
    if (this.cfg && this.cfg.trigger === "oneshot") {
      this.activeMidis.delete(midi);
      this.onVoicesChange && this.onVoicesChange(this.activeMidis);
      return; // one-shots ring out to the end
    }
    if (this.cfg.mode === "tape") this._tapeOff(midi);
    else this._grainOff(midi);
    this.activeMidis.delete(midi);
    this.onVoicesChange && this.onVoicesChange(this.activeMidis);
  }

  allNotesOff() {
    [...this.activeMidis].forEach((m) => this.noteOff(m));
  }

  keyGain(midi) { return this.keyGains[midi] == null ? 1 : this.keyGains[midi]; }

  setKeyGain(midi, value) {
    this.keyGains[midi] = value;
    // Live-follow: currently sustaining tape voices of this key move to the
    // new gain (their stored base already includes the velocity).
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.tapeVoices.forEach((v) => {
      if (v.midi !== midi || v.stopAt) return;
      const target = Math.max(0.0001, value * (v.baseVelocity || 1));
      try { v.gainNode.gain.setTargetAtTime(target, t, 0.03); } catch (_) {}
    });
  }

  _voiceGain(midi, velocity) {
    return Math.max(0, Math.min(4, this.keyGain(midi))) * velocity;
  }

  /* ------------------------------------------------------------- tape mode */

  _tapeOn(midi, velocity) {
    const ratio = Math.pow(2, (midi - this.cfg.root) / 12);
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    src.playbackRate.value = ratio;
    const g = this.ctx.createGain();
    const gv = this._voiceGain(midi, velocity);
    g.gain.value = gv;
    src.connect(g);
    g.connect(this.master);
    const sr = this.buffer.sampleRate;
    const off = this.regionStartAbs || 0;
    const loop = this.cfg.trigger === "loop";
    const t = this.ctx.currentTime;
    if (loop) {
      src.loop = true;
      src.loopStart = off + this.cfg.loopStart;
      src.loopEnd = off + this.cfg.loopEnd;
    }
    src.onended = () => {
      try { g.disconnect(); } catch (_) {}
      this.tapeVoices = this.tapeVoices.filter((x) => x.src !== src);
    };
    const voice = { midi, src, gain: gv, baseVelocity: velocity, gainNode: g, stopAt: null };
    // Short attack ramp to avoid clicks.
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(Math.max(0.0002, gv), t + 0.008);
    src.start(t, off);
    if (!loop) {
      const regionDur = this.region.length / sr;
      src.stop(t + regionDur / ratio + 0.05);
    }
    this.tapeVoices.push(voice);
    this._stealIfNeeded();
  }

  _tapeOff(midi) {
    // Release every voice of this midi (loop voices) with a fast fade.
    const t = this.ctx.currentTime;
    this.tapeVoices.filter((v) => v.midi === midi).forEach((v) => {
      if (v.stopAt) return;
      v.stopAt = t + 0.1;
      v.gainNode.gain.cancelScheduledValues(t);
      v.gainNode.gain.setValueAtTime(v.gainNode.gain.value, t);
      v.gainNode.gain.exponentialRampToValueAtTime(0.0001, t + 0.09);
      try { v.src.stop(t + 0.12); } catch (_) {}
    });
  }

  _stealIfNeeded() {
    const count = this.tapeVoices.length + this.grainVoices.length;
    if (count <= this.maxVoices) return;
    // Steal the oldest released tape voice, else oldest tape voice.
    const rel = this.tapeVoices.filter((v) => v.stopAt).sort((a, b) => a.stopAt - b.stopAt);
    if (rel.length) { try { rel[0].src.stop(); } catch (_) {} return; }
    if (this.tapeVoices.length) { try { this.tapeVoices[0].src.stop(); } catch (_) {} return; }
    if (this.grainVoices.length) this.grainVoices.shift();
  }

  /* ------------------------------------------------------------ grain mode */

  _grainOn(midi, velocity) {
    const G = Math.max(64, Math.round(this.grainMs * this.procSr));
    const voice = {
      midi,
      ratio: Math.pow(2, (midi - this.cfg.root) / 12),
      gain: this._voiceGain(midi, velocity),
      loop: this.cfg.trigger === "loop",
      loop0: this.cfg.loop0,                 // processor-rate frames
      loop1: this.cfg.loop1,
      nProc: this.region.length / this.buffer.sampleRate * this.procSr,
      center: G / 2,
      age: 0,
      released: false,
      holdLen: 0,
      attack: 0.006 * this.procSr,
      release: 0.09 * this.procSr,
    };
    this.grainVoices.push(voice);
    this._stealIfNeeded();
  }

  _grainOff(midi) {
    this.grainVoices.filter((v) => v.midi === midi && !v.released).forEach((v) => {
      v.released = true;
      v.holdLen = v.age;
    });
  }

  _readRegion(voice, procIdx) {
    // procIdx: position in processor-rate frames; wrap the loop there first,
    // then convert to source-rate frames to index the region Float32Array.
    if (voice.loop && procIdx >= voice.loop1) {
      const span = voice.loop1 - voice.loop0;
      if (span > 0) procIdx = voice.loop0 + ((procIdx - voice.loop0) % span);
    }
    const idx = procIdx * (this.buffer.sampleRate / this.procSr);
    const s = this.region;
    const n = s.length;
    if (idx <= 0) return s[0];
    const i0 = idx | 0;
    if (i0 >= n - 1) return s[n - 1];
    const f = idx - i0;
    return s[i0] * (1 - f) + s[i0 + 1] * f;
  }

  _renderGrains(out) {
    const M = out.length;
    const voices = this.grainVoices;
    if (!voices.length) return;
    const G = Math.max(64, Math.round(this.grainMs * this.procSr));
    const H = G >> 1;
    const half = G / 2;
    const TWO_PI = 2 * Math.PI;
    for (let vi = voices.length - 1; vi >= 0; vi--) {
      const v = voices[vi];
      const ratio = v.ratio;
      const reach = Math.ceil(half / ratio);
      let done = false;
      for (let frame = 0; frame < M; frame++) {
        const outPos = v.age + frame; // output frames since voice onset
        // Grain centres live at k*H on both the output-placement and the
        // source-centre timeline; iterate centres whose window covers frame.
        let acc = 0, wsum = 0;
        const kLo = Math.max(1, Math.floor((outPos - reach) / H));
        const kHi = Math.ceil((outPos + reach) / H);
        for (let k = kLo; k <= kHi; k++) {
          const oc = k * H;
          const j = outPos - oc;
          if (j < -reach || j > reach) continue;
          const f = ratio * j;                  // source offset within grain
          const ph = (f + half) / G;            // Hann window phase
          if (ph <= 0 || ph >= 1) continue;
          let srcCenter = k * H;
          if (v.loop && srcCenter >= v.loop1) {
            const span = v.loop1 - v.loop0;
            if (span > 0) srcCenter = v.loop0 + ((srcCenter - v.loop0) % span);
          }
          // One-shot: nothing beyond the region tail.
          if (!v.loop && srcCenter - half > v.nProc) continue;
          const w = 0.5 - 0.5 * Math.cos(TWO_PI * ph);
          acc += w * this._readRegion(v, srcCenter + f);
          wsum += w;
        }
        if (wsum > 1e-6) {
          acc /= wsum;
          // Attack/release envelope.
          let env = 1;
          if (outPos < v.attack) env = outPos / v.attack;
          if (v.released) {
            const rel = outPos - v.holdLen;
            if (rel >= v.release) {
              acc = 0;
              if (outPos > v.holdLen + v.release + G) done = true;
            } else env *= 1 - rel / v.release;
          } else if (!v.loop) {
            // Natural taper as the read head leaves a one-shot region.
            const remain = v.nProc - outPos;
            if (remain < v.release && remain > 0) env *= remain / v.release;
            if (remain <= -G) done = true;
          }
          out[frame] += acc * env * v.gain;
        }
      }
      v.age += M;
      if (done || (!v.loop && v.age > v.nProc + G * 2)) {
        voices.splice(vi, 1);
      }
    }
  }

  /* --------------------------------------------------------------- meters */
}

/* ========================================================================= */
/* WAV encode / upload helpers                                              */
/* ========================================================================= */

function encodeWavPcm16(channels, sr) {
  const n = channels[0].length;
  const ch = channels.length;
  const buf = new ArrayBuffer(44 + n * ch * 2);
  const v = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + n * ch * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, ch, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * ch * 2, true); v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true);
  ws(36, "data"); v.setUint32(40, n * ch * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < ch; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* Tap the master bus into a growing mono Float32 list for live recording. */
class MasterRecorder {
  constructor(engine) {
    this.engine = engine;
    this.chunks = [];
    this.frames = 0;
    this.node = null;
    this.startTime = 0;
    this.events = [];
  }

  start() {
    const ctx = this.engine.ctx;
    this.node = ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.frames = 0;
    this.startTime = ctx.currentTime;
    this.node.onaudioprocess = (e) => {
      const d = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(d));
      this.frames += d.length;
    };
    // Tap master post-gain: master -> recorder node (processor must reach destination)
    this.engine.master.connect(this.node);
    this.node.connect(ctx.destination);
  }

  stop() {
    return new Promise((resolve) => {
      // Capture one more tail quantum so the last notes ring out.
      setTimeout(() => {
        try { this.engine.master.disconnect(this.node); } catch (_) {}
        try { this.node.disconnect(); } catch (_) {}
        const all = new Float32Array(this.frames);
        let off = 0;
        for (const c of this.chunks) { all.set(c, off); off += c.length; }
        resolve({ samples: all, sr: this.engine.ctx.sampleRate, duration: this.frames / this.engine.ctx.sampleRate });
      }, 120);
    });
  }
}
