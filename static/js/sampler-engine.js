/* sampler-engine.js — Main-thread controller for the AudioWorklet sampler.
 *
 * Responsibilities:
 *   * decode the library WAV (decodeAudioData), resample it to the audio
 *     context rate and slice the configured region once ("load sample");
 *   * map MIDI keys -> playback rate and per-key volume, track held keys and
 *     recorded note events with audio-clock timestamps;
 *   * forward envelope / timbre / loop / mode parameter changes live.
 */

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function midiName(m) { return NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1); }
function midiRate(semi) { return Math.pow(2, semi / 12); }

/* Scale intervals from the tonic (semitone offsets in the octave). */
const SCALES = {
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic_major: [0, 2, 4, 7, 9],
  pentatonic_minor: [0, 3, 5, 7, 10],
  blues: [0, 3, 5, 6, 7, 10],
};

/* Linear-interpolation resampler (keeps the worklet perfectly in tune with
 * its own AudioContext sample rate; same algorithm as backend dsp.py). */
function resampleLinear(src, srcSr, dstSr) {
  if (srcSr === dstSr) return src;
  const ratio = srcSr / dstSr;
  const nOut = Math.round(src.length / ratio);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const pos = i * ratio;
    const i0 = pos | 0;
    const f = pos - i0;
    out[i] = i0 + 1 < src.length ? src[i0] * (1 - f) + src[i0 + 1] * f : src[Math.min(i0, src.length - 1)];
  }
  return out;
}

class SamplerEngine {
  constructor() {
    this.ctx = null;
    this.node = null;
    this.loaded = false;
    this.held = new Map();       // midi -> current event {start}
    this.volumes = new Map();    // midi -> 0..1.5
    this.events = [];            // recorded performance
    this.recording = false;
    this.cfg = {
      mode: "varispeed", loopOn: false,
      attack: 0.005, decay: 0.0, sustain: 1.0, release: 0.15,
      cutoff: 0, master: 0.9, loopStart: 0, loopEnd: 0, loopXfade: 0.02,
      rootMidi: 60,
    };
  }

  async init() {
    if (this.ctx) {
      if (this.ctx.state === "suspended") await this.ctx.resume();
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx();
    await this.ctx.audioWorklet.addModule("/static/js/sampler-worklet.js");
    this.node = new AudioWorkletNode(this.ctx, "sampler-processor");
    this.node.connect(this.ctx.destination);
  }

  get now() { return this.ctx ? this.ctx.currentTime : 0; }

  /* Decode the file, resample to the context rate and transfer the sliced
   * region into the worklet (only the region is sent — keeps key latency low
   * even for huge source files). */
  async loadSample(fileId, regionStart, regionEnd) {
    await this.init();
    const res = await fetch(API.fileUrl(fileId));
    const arr = await res.arrayBuffer();
    const decoded = await this.ctx.decodeAudioData(arr.slice(0));
    const sr = this.ctx.sampleRate;
    const a = Math.max(0, Math.round(regionStart * decoded.sampleRate));
    const b = Math.min(decoded.length, Math.round(regionEnd * decoded.sampleRate));
    const chans = [];
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const full = decoded.getChannelData(c);
      const slice = full.slice(a, Math.max(a + 1, b));
      chans.push(resampleLinear(slice, decoded.sampleRate, sr));
    }
    this.node.port.postMessage(
      {
        type: "load",
        buffer: chans,
        sr,
        loop: [this.cfg.loopStart, this.cfg.loopEnd],
        xfade: this.cfg.loopXfade,
        mode: this.cfg.mode,
        loopOn: this.cfg.loopOn,
      },
      chans.map((c) => c.buffer),
    );
    // Make sure envelope/timbre params match the current UI after a (re)build.
    this.updateParams();
    this.node.port.postMessage({ type: "offAll" });
    this.held.clear();
    this.loaded = true;
    return { frames: chans[0].length, sr, channels: chans.length,
             duration: chans[0].length / sr };
  }

  updateLoop() {
    if (!this.loaded) return;
    this.node.port.postMessage({
      type: "loop",
      loop: [this.cfg.loopStart, this.cfg.loopEnd],
      xfade: this.cfg.loopXfade,
      mode: this.cfg.mode,
      loopOn: this.cfg.loopOn,
    });
  }

  updateParams() {
    if (!this.node) return;
    this.node.port.postMessage({ type: "params", params: {
      mode: this.cfg.mode,
      attack: this.cfg.attack, decay: this.cfg.decay,
      sustain: this.cfg.sustain, release: this.cfg.release,
      cutoff: this.cfg.cutoff, master: this.cfg.master,
    } });
  }

  noteOn(midi) {
    if (!this.loaded) return;
    if (this.held.has(midi)) return;
    const volume = this.volumes.has(midi) ? this.volumes.get(midi) : 1.0;
    const semi = midi - this.cfg.rootMidi;
    this.node.port.postMessage({
      type: "note", midi, rate: midiRate(semi), volume,
    });
    const ev = { midi, start: this.now, volume };
    this.held.set(midi, ev);
    if (this.recording) this.events.push(ev);
  }

  noteOff(midi) {
    const ev = this.held.get(midi);
    if (!ev) return;
    this.node.port.postMessage({ type: "off", midi });
    this.held.delete(midi);
    if (this.recording) ev.duration = Math.max(0.02, this.now - ev.start);
  }

  allNotesOff() {
    if (!this.node) return;
    this.node.port.postMessage({ type: "offAll" });
    const now = this.now;
    for (const [midi, ev] of this.held) {
      if (this.recording) ev.duration = Math.max(0.02, now - ev.start);
    }
    this.held.clear();
  }

  startRecording() {
    // Clear any notes still held (their events, if any, belong to the previous
    // take), then stamp the recording origin — context time of t=0 in the
    // rendered file.
    if (this.node) this.node.port.postMessage({ type: "offAll" });
    this.held.clear();
    this.events = [];
    this._recOrigin = this.now;
    this.recording = true;
  }

  stopRecording() {
    // Stamp durations on any notes still held before flipping the flag, so a
    // note released together with "stop" still ends up in the performance.
    const now = this.now;
    for (const ev of this.held.values()) {
      ev.duration = Math.max(0.02, now - ev.start);
    }
    if (this.node) this.node.port.postMessage({ type: "offAll" });
    this.held.clear();
    this.recording = false;
    const origin = this._recOrigin || 0;
    return this.events
      .filter((e) => e.duration > 0)
      .map((e) => ({ midi: e.midi, start: Math.max(0, e.start - origin),
                     duration: e.duration, volume: e.volume }));
  }

  pendingCount() { return this.held.size; }
}
