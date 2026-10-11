// Inference worker: loads the ONNX models (cached with the Cache API), then runs TurnStream on
// every 80 ms stereo block the AudioWorklet sends, and posts per-frame scores back to the page.
import * as ort from './ort/ort.webgpu.bundle.min.mjs';
import { TurnStream } from './turn_stream.js';

ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = 'turn-playground-v1';
const MAX_BACKLOG = 40;          // blocks (3.2 s) of unprocessed audio before we give up
let ts = null, queue = [], busy = false, stopped = true, blocks = 0;
const levels = new Array(64);

async function fetchCached(url, label) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache API */ }
  const hit = cache && await cache.match(url);
  if (hit) {
    postMessage({ type: 'progress', label, done: 1, total: 1, cached: true });
    return new Uint8Array(await hit.arrayBuffer());
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not download ${url} (HTTP ${r.status})`);
  const total = +r.headers.get('content-length') || 0;
  const reader = r.body.getReader();
  const parts = [];
  let done = 0, last = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    parts.push(value);
    done += value.length;
    if (performance.now() - last > 150) { last = performance.now(); postMessage({ type: 'progress', label, done, total }); }
  }
  const out = new Uint8Array(done);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  postMessage({ type: 'progress', label, done, total: done });
  if (cache) cache.put(url, new Response(out.slice())).catch(() => {});
  return out;
}

async function init({ ep, encoder, frames }) {
  const enc = await fetchCached(encoder, 'encoder');
  const head = await fetchCached('models/head.onnx', 'head');
  postMessage({ type: 'status', text: `Compiling the model for ${ep === 'webgpu' ? 'WebGPU' : 'CPU'}…` });
  ts = await TurnStream.create(ort, { encoder: enc, head, frames, ep });
  // warm-up: one silent chunk so the first real call does not pay shader compilation
  for (let i = 0; i < frames; i++) await ts.push(new Float32Array(1280), new Float32Array(1280));
  ts.reset();
  postMessage({ type: 'ready', ep, frames, threads: ort.env.wasm.numThreads });
}

const rms = (x) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / x.length); };

async function pump() {
  if (busy) return;
  busy = true;
  while (queue.length && !stopped) {
    if (queue.length > MAX_BACKLOG) {
      stopped = true;
      queue = [];
      postMessage({ type: 'overload' });
      break;
    }
    const blk = queue.shift();
    const s = performance.now();
    levels[blocks++ % 64] = [rms(blk.a), rms(blk.b)];
    const outs = await ts.push(blk.a, blk.b);
    if (!outs.length) continue;
    // frame t's newest audio is block t - 1 (0-based), so show that block's level with it
    const lv = (t) => (t >= 1 && t <= blocks ? levels[(t - 1) % 64] : [0, 0]);
    postMessage({ type: 'frames', outs: outs.map((o) => ({ t: o.t, eot: o.eot, int: o.int, silent: o.silent, level: lv(o.t) })),
                  ms: performance.now() - s, lag: performance.now() - blk.recv, backlog: queue.length });
  }
  busy = false;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m);
    else if (m.type === 'start') {
      ts.reset();
      queue = [];
      blocks = 0;
      stopped = false;
      m.port.onmessage = (ev) => {
        if (stopped) return;
        queue.push({ ...ev.data, recv: performance.now() });
        if (queue.length > MAX_BACKLOG) { stopped = true; queue = []; postMessage({ type: 'overload' }); return; }
        pump();
      };
    } else if (m.type === 'stop') {
      stopped = true;
      queue = [];
    }
  } catch (err) {
    postMessage({ type: 'error', text: err && err.message ? err.message : String(err) });
  }
};
