// AudioWorklet: collects the two input channels (16 kHz context) into 1280-sample (80 ms) blocks
// and sends them straight to the inference worker over a MessagePort handed in by app.js.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.n = 0;
    this.a = new Float32Array(1280);
    this.b = new Float32Array(1280);
    this.out = null;
    this.from = Infinity;      // first sample frame to keep, so files start exactly on a block edge
    this.port.onmessage = (e) => { this.out = e.data.port; this.from = e.data.startFrame; this.n = 0; };
  }

  process(inputs) {
    const inp = inputs[0] || [];
    const L = inp[0], R = inp[1];
    const len = L ? L.length : 128;
    for (let i = Math.max(0, Math.min(len, this.from - currentFrame)); i < len; i++) {
      this.a[this.n] = L ? L[i] : 0;
      this.b[this.n] = R ? R[i] : 0;
      if (++this.n === 1280) {
        if (this.out) this.out.postMessage({ a: this.a, b: this.b, at: currentTime }, [this.a.buffer, this.b.buffer]);
        this.a = new Float32Array(1280);
        this.b = new Float32Array(1280);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture', Capture);
