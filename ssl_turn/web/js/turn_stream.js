// Streaming ssl_turn in the browser: FastConformer streaming encoder + turn head with
// onnxruntime-web (WebGPU or WASM). See ssl_turn/web/README.md.
//
//   const ts = await TurnStream.create(ort, {encoder: 'encoder_k2_w16.onnx', frames: 2, head: 'head.onnx', ep: 'webgpu'});
//   for each 80 ms block: const outs = await ts.push(user, agent)   // two Float32Array(1280), 16 kHz
//   -> [{t, eot: [c0, c1], int: [c0, c1], silent: [c0, c1], post}, ...]  (score tracks per channel)
//
// `frames` (K) must match the encoder file: the encoder runs once per K blocks and returns K
// frames. Frame t's feature needs audio only up to sample 1280 t - 1024, so with K = 1 frame t is
// ready as soon as block t - 1 has arrived; with K > 1 the earlier frames of a chunk wait for the
// last block of the chunk. The first push also returns frame 0 (its feature is zeros).

export const FRAME = 1280;           // 80 ms at 16 kHz
const LAYERS = 17, CACHE = 70, D = 512, CONV = 8, FEAT = 1024;
const HEAD_SHAPE = [6, 2, 4, 249, 48];   // [2 x head layers, channels, heads, W - 1, head dim] (r019 head)
const HEAD_W = 250;
const STATE = ['cache_len', ...[...Array(LAYERS).keys()].map((i) => `cache_ch_${i}`),
  ...[...Array(LAYERS).keys()].map((i) => `cache_time_${i}`)];
const win = (K) => 3073 + 1280 * (K - 1);   // stream_encoder.win(K)

// Pick an encoder variant for this browser: fp16 math needs WebGPU with shader-f16; w16 (fp16
// weights, fp32 math) runs anywhere; int8 is the fastest on WASM.
export async function pickVariant() {
  const a = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!a) return { ep: 'wasm', variant: 'int8' };
  return { ep: 'webgpu', variant: a.features.has('shader-f16') ? 'fp16' : 'w16' };
}

export class TurnStream {
  // ep: encoder backend ('webgpu' or 'wasm'). The head (3M params) runs on WASM by default.
  static async create(ort, { encoder, head, frames = 1, ep = 'webgpu', headEp = 'wasm' }) {
    const self = new TurnStream();
    self.ort = ort;
    self.K = frames;
    const base = { executionProviders: [ep], graphOptimizationLevel: 'all' };
    // Caches stay on the GPU between steps; only the features are read back.
    self.enc = await ort.InferenceSession.create(encoder, ep === 'webgpu'
      ? { ...base, preferredOutputLocation: Object.fromEntries(STATE.slice(1).map((n) => [n + '_out', 'gpu-buffer'])) }
      : base);
    self.head = await ort.InferenceSession.create(head, { executionProviders: [headEp], graphOptimizationLevel: 'all' });
    self.timing = { enc: 0, head: 0 };
    self.reset();
    return self;
  }

  reset() {
    const T = this.ort.Tensor;
    for (const s of Object.values(this.state || {})) if (s.location === 'gpu-buffer') s.dispose();
    this.back = win(this.K) + 1024;     // samples kept per channel: the window plus the 1024 it skips
    this.buf = [new Float32Array(this.back), new Float32Array(this.back)];
    this.pending = 0;                   // blocks received since the last encoder call
    this.t = 1;                         // first frame of the next chunk
    this.state = { cache_len: new T('int64', new BigInt64Array(2), [2]) };
    for (let i = 0; i < LAYERS; i++) {
      this.state[`cache_ch_${i}`] = new T('float32', new Float32Array(2 * CACHE * D), [2, CACHE, D]);
      this.state[`cache_time_${i}`] = new T('float32', new Float32Array(2 * D * CONV), [2, D, CONV]);
    }
    const n = HEAD_SHAPE.reduce((a, b) => a * b);
    this.kc = new T('float32', new Float32Array(n), HEAD_SHAPE);
    this.vc = new T('float32', new Float32Array(n), HEAD_SHAPE);
    this.started = false;
  }

  // feat: Float32Array [2, K, 1024] for frames t0 .. t0 + K - 1
  async headStep(feat, t0, K) {
    const T = this.ort.Tensor;
    const s = performance.now();
    const n = new T('int64', BigInt64Array.from([BigInt(Math.min(t0, HEAD_W - 1))]), [1]);
    const o = await this.head.run({ feat: new T('float32', feat, [2, K, FEAT]), n, kcache: this.kc, vcache: this.vc });
    this.kc = o.kcache_out;
    this.vc = o.vcache_out;
    const [eot, intr, silent, post] = await Promise.all(['eot', 'int', 'silent', 'post'].map((k) => o[k].getData()));
    this.timing.head = performance.now() - s;
    const out = [];
    for (let j = 0; j < K; j++) {
      out.push({ t: t0 + j, eot: [eot[2 * j], eot[2 * j + 1]], int: [intr[2 * j], intr[2 * j + 1]],
                 silent: [silent[2 * j], silent[2 * j + 1]], post: post.subarray(16 * j, 16 * j + 16) });
    }
    return out;
  }

  // Feed the next 80 ms block per channel; returns the frames that became available (maybe none).
  async push(ch0, ch1) {
    const T = this.ort.Tensor;
    let out = [];
    if (!this.started) {
      this.started = true;
      out = await this.headStep(new Float32Array(2 * FEAT), 0, 1);
    }
    for (const [c, x] of [[0, ch0], [1, ch1]]) {
      this.buf[c].copyWithin(0, FRAME);
      this.buf[c].set(x, this.back - FRAME);
    }
    if (++this.pending < this.K) return out;
    this.pending = 0;
    const K = this.K, W = win(K), s = performance.now();
    const audio = new Float32Array(2 * W);
    audio.set(this.buf[0].subarray(0, W), 0);
    audio.set(this.buf[1].subarray(0, W), W);
    const o = await this.enc.run({
      audio: new T('float32', audio, [2, W]),
      t: new T('int64', BigInt64Array.from([BigInt(this.t)]), [1]),
      ...this.state,
    });
    for (const name of STATE) {
      const prev = this.state[name];
      if (prev.location === 'gpu-buffer') prev.dispose();
      this.state[name] = o[name + '_out'];
    }
    const feat = await o.feat.getData(true);           // [2, K, 1024]; true releases a GPU buffer
    this.timing.enc = performance.now() - s;
    out = out.concat(await this.headStep(feat, this.t, K));
    this.t += K;
    return out;
  }
}
