// AudioWorklet: mixes input to mono, resamples to 24 kHz and posts 16-bit PCM chunks (~100 ms each)
// together with an RMS level for the on-screen meter.
class Pcm16Writer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    const target = opts.targetRate || 24000;
    this.ratio = sampleRate / target;
    this.chunk = Math.round((target * (opts.chunkMs || 100)) / 1000);
    this.buf = new Int16Array(this.chunk);
    this.len = 0;
    this.sumSq = 0;
    this.t = 0; // read position of the next output sample, relative to the current block
    this.prev = 0; // last (filtered) input sample of the previous block
    this.lp = 0; // previous raw sample, for the anti-alias filter
  }

  push(sample) {
    const v = Math.max(-1, Math.min(1, sample));
    this.buf[this.len++] = v < 0 ? v * 0x8000 : v * 0x7fff;
    this.sumSq += v * v;
    if (this.len === this.chunk) {
      const out = this.buf;
      this.port.postMessage({ pcm: out.buffer, level: Math.sqrt(this.sumSq / this.chunk) }, [out.buffer]);
      this.buf = new Int16Array(this.chunk);
      this.len = 0;
      this.sumSq = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (const channel of input) {
      for (let i = 0; i < n; i++) mono[i] += channel[i] / input.length;
    }
    if (this.ratio > 1.2) {
      // Two-tap low-pass before downsampling keeps high frequencies from folding into speech.
      for (let i = 0; i < n; i++) {
        const raw = mono[i];
        mono[i] = 0.5 * (raw + this.lp);
        this.lp = raw;
      }
    }
    let t = this.t;
    while (t < n - 1) {
      const i = Math.floor(t);
      const f = t - i;
      const a = i < 0 ? this.prev : mono[i];
      this.push(a + (mono[i + 1] - a) * f);
      t += this.ratio;
    }
    this.t = t - n;
    this.prev = mono[n - 1];
    return true;
  }
}

registerProcessor('pcm16-writer', Pcm16Writer);
