// Turn-taking playground: two speakers (mic or audio file each, or one stereo file), streamed
// through the turn model in a worker, with the TurnBench commit policy applied live.
const FPS = 12.5, SPAN = 20;               // timeline shows the last 20 s
const NF = Math.round(SPAN * FPS);
const $ = (id) => document.getElementById(id);
const COL = { level: '#8a8f98', speak: 'rgba(80,160,110,0.18)', eot: '#3a7bd5', int: '#e0742a' };

// --- commit policy (turnbench.sweep.commit_events + ssl_turn/pipeline/score.commit) ----------
// Rising edge above theta fires (refractory counts committed fires only); EOT also re-commits
// once if the score stays above theta for 1 s after a fire.
class Committer {
  constructor(refractory_s, recommit_s) {
    this.refr = Math.max(1, Math.round(refractory_s * FPS));
    this.rec = recommit_s ? Math.round(recommit_s * FPS) : 0;
    this.reset();
  }
  reset() { this.prev = false; this.last = -1e9; this.runFrom = -1; }
  // returns 'fire' | 'recommit' | null for frame i
  step(i, p, theta) {
    const above = p > theta;
    let ev = null;
    if (above && !this.prev && i - this.last >= this.refr) { ev = 'fire'; this.last = i; this.runFrom = i; }
    if (!above) this.runFrom = -1;
    else if (this.rec && this.runFrom >= 0 && i - this.runFrom === this.rec) { ev = ev || 'recommit'; this.runFrom = -1; }
    this.prev = above;
    return ev;
  }
}

// --- state ---------------------------------------------------------------------------------
let worker = null, ready = null, framesPerCall = 2, ctx = null, graph = null, running = false;
let hist = null, events = null, commits = null, lastT = -1, stats = null;
const spk = [
  { name: 'A', src: 'mic', file: null },
  { name: 'B', src: 'file', file: null },
];
let mode = 'two', stereoFile = null, bgFile = null;

function theta() { return { eot: +$('th-eot').value, int: +$('th-int').value }; }

function resetHistory() {
  hist = Array.from({ length: 2 }, () => ({ level: new Float32Array(NF), eot: new Float32Array(NF), int: new Float32Array(NF), speak: new Float32Array(NF) }));
  events = [];        // {t, c, kind: 'eot'|'int', re}
  commits = [0, 1].map(() => ({ eot: new Committer(0.5, 1.0), int: new Committer(0.5, null) }));
  lastT = -1;
  stats = { calls: 0, ms: 0, lag: 0, backlog: 0, counts: [{ eot: 0, int: 0 }, { eot: 0, int: 0 }] };
}

// --- worker / model --------------------------------------------------------------------------
async function detectBackend() {
  if (!navigator.gpu) return 'wasm';
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  // a software adapter (no usable GPU) is much slower than the WASM backend
  return a && !(a.info && a.info.isFallbackAdapter) && !a.isFallbackAdapter ? 'webgpu' : 'wasm';
}

async function loadModel() {
  if (ready) return ready;
  ready = (async () => {
    let ep = $('backend').value;
    if (ep === 'auto') ep = await detectBackend();
    const frames = ep === 'webgpu' ? 2 : 4;
    const encoder = `models/encoder_k${frames}_w16.onnx`;
    worker = new Worker('worker.js', { type: 'module' });
    worker.onmessage = onWorker;
    setStatus(`Downloading the model (about 225 MB, cached after the first visit)…`);
    $('load').disabled = true;
    $('backend').disabled = true;
    await new Promise((res, rej) => {
      worker._ready = res; worker._fail = rej;
      worker.postMessage({ type: 'init', ep, encoder: new URL(encoder, location.href).href, frames });
    });
    return ep;
  })();
  ready.catch((e) => { ready = null; $('load').disabled = false; $('backend').disabled = false; setStatus(`Could not load: ${e.message || e}`, true); });
  return ready;
}

const progress = {};
function onWorker(e) {
  const m = e.data;
  if (m.type === 'progress') {
    progress[m.label] = m;
    const done = Object.values(progress).reduce((a, p) => a + p.done, 0);
    const total = Object.values(progress).reduce((a, p) => a + (p.total || p.done), 0);
    $('bar').style.width = `${total ? (100 * done / total) : 0}%`;
    setStatus(m.cached ? 'Loading the model from the browser cache…' : `Downloading the model: ${(done / 1e6).toFixed(0)} MB${total ? ` of ${(total / 1e6).toFixed(0)} MB` : ''}`);
  } else if (m.type === 'status') setStatus(m.text);
  else if (m.type === 'ready') {
    $('bar').style.width = '100%';
    const be = m.ep === 'webgpu' ? 'WebGPU' : `CPU (WASM, ${m.threads} thread${m.threads > 1 ? 's' : ''})`;
    framesPerCall = m.frames;
    $('st-backend').textContent = `${be}, ${m.frames} frames per call`;
    setStatus('Model ready. Pick the two sources and press Start.');
    $('start').disabled = false;
    worker._ready(m.ep);
  } else if (m.type === 'error') {
    if (worker._fail) worker._fail(new Error(m.text));
    setStatus(`Error: ${m.text}`, true);
    stop();
  } else if (m.type === 'overload') {
    setStatus('This device cannot keep up in real time with this backend, so the stream was stopped. Try the other backend.', true);
    stop(false);
  } else if (m.type === 'frames') onFrames(m);
}

const debug = new URLSearchParams(location.search).has('debug');
const trace = [];
window.__pg = () => ({ lastT, events, stats, trace, running });

function onFrames(m) {
  if (debug) for (const o of m.outs) trace.push([o.t, ...o.eot, ...o.int]);
  stats.ms = stats.calls++ ? 0.8 * stats.ms + 0.2 * m.ms : m.ms;
  stats.lag = stats.calls > 1 ? 0.8 * stats.lag + 0.2 * m.lag : m.lag;
  stats.backlog = m.backlog;
  const th = theta();
  for (const o of m.outs) {
    const i = o.t % NF;
    for (let c = 0; c < 2; c++) {
      const h = hist[c];
      h.eot[i] = o.eot[c]; h.int[i] = o.int[c]; h.speak[i] = 1 - o.silent[c];
      h.level[i] = o.level[c];
      for (const kind of ['eot', 'int']) {
        const ev = commits[c][kind].step(o.t, o[kind][c], th[kind]);
        if (ev) {
          events.push({ t: o.t, c, kind, re: ev === 'recommit' });
          stats.counts[c][kind]++;
          flash(c, kind);
        }
      }
    }
    lastT = o.t;
  }
  while (events.length && events[0].t < lastT - NF) events.shift();
}

// --- audio graph -----------------------------------------------------------------------------
async function decode(file) {
  const buf = await file.arrayBuffer();
  return await ctx.decodeAudioData(buf);
}

async function micStream() {
  const proc = $('mic-proc').checked;
  return navigator.mediaDevices.getUserMedia({ audio: {
    channelCount: 1, echoCancellation: true, noiseSuppression: proc, autoGainControl: proc,
  } });
}

async function buildGraph() {
  ctx = new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });
  await ctx.audioWorklet.addModule('capture.js');
  const merger = ctx.createChannelMerger(2);
  const node = new AudioWorkletNode(ctx, 'capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete' });
  const mute = ctx.createGain();
  mute.gain.value = 0;
  merger.connect(node).connect(mute).connect(ctx.destination);
  const g = { merger, node, sources: [], stream: null, files: 0, ended: 0 };
  const monitor = $('monitor').checked;

  const playFile = (audioBuf, dests) => {
    const s = ctx.createBufferSource();
    s.buffer = audioBuf;
    for (const d of dests) s.connect(...d);
    g.sources.push(s);
    g.files++;
    s.onended = () => { if (++g.ended >= g.files && !g.stream && running) setTimeout(() => stop(), 1200); };
    return s;
  };
  const toChannel = (c) => [merger, 0, c];

  if (mode === 'stereo') {
    if (!stereoFile) throw new Error('Choose a stereo file first.');
    const b = await decode(stereoFile);
    const split = ctx.createChannelSplitter(2);
    const s = playFile(b, [[split]]);
    split.connect(merger, 0, 0);
    split.connect(merger, b.numberOfChannels > 1 ? 1 : 0, 1);
    if (monitor) s.connect(ctx.destination);
  } else {
    if (spk[0].src === 'mic' && spk[1].src === 'mic') throw new Error('Only one speaker can use the microphone.');
    for (let c = 0; c < 2; c++) {
      const sp = spk[c];
      if (sp.src === 'mic') {
        g.stream = await micStream();
        ctx.createMediaStreamSource(g.stream).connect(merger, 0, c);
      } else if (sp.src === 'file') {
        if (!sp.file) throw new Error(`Choose an audio file for speaker ${sp.name}, or set it to Silent.`);
        const s = playFile(await decode(sp.file), [toChannel(c)]);
        if (monitor) s.connect(ctx.destination);
      }
    }
  }
  if (bgFile) {
    // background audio mixed into speaker A's channel (and heard, so it also reaches a live mic)
    const b = await decode(bgFile);
    const s = ctx.createBufferSource();
    s.buffer = b;
    s.loop = true;
    const gain = ctx.createGain();
    gain.gain.value = +$('bg-gain').value;
    g.bgGain = gain;
    s.connect(gain).connect(merger, 0, 0);
    if (monitor) gain.connect(ctx.destination);
    g.sources.push(s);
  }
  return g;
}

async function start() {
  try {
    $('start').disabled = true;
    await loadModel();
    resetHistory();
    graph = await buildGraph();
    const ch = new MessageChannel();
    // capture and file playback both start at t0, so a file's first sample is the stream's first
    const t0 = ctx.currentTime + 0.15;
    worker.postMessage({ type: 'start', port: ch.port2 }, [ch.port2]);
    graph.node.port.postMessage({ port: ch.port1, startFrame: Math.round(t0 * ctx.sampleRate) }, [ch.port1]);
    running = true;
    for (const s of graph.sources) s.start(t0);
    $('stop').disabled = false;
    lockInputs(true);
    setStatus(graph.stream ? 'Listening. Talk into the microphone.' : 'Streaming the files…');
  } catch (e) {
    setStatus(e.message || String(e), true);
    stop();
  }
}

function stop(keepMsg = true) {
  if (worker) worker.postMessage({ type: 'stop' });
  if (graph) {
    for (const s of graph.sources) { s.onended = null; try { s.stop(); } catch { /* not started */ } }
    if (graph.stream) graph.stream.getTracks().forEach((t) => t.stop());
    graph = null;
  }
  if (ctx) { ctx.close(); ctx = null; }
  const was = running;
  running = false;
  $('start').disabled = !ready;
  $('stop').disabled = true;
  lockInputs(false);
  if (was && keepMsg) setStatus('Stopped. Press Start to run again.');
}

function lockInputs(on) {
  for (const el of document.querySelectorAll('[data-lock]')) el.disabled = on;
}

// --- UI --------------------------------------------------------------------------------------
function setStatus(text, err = false) {
  $('status').textContent = text;
  $('status').classList.toggle('err', err);
}

const chipTimers = {};
function flash(c, kind) {
  const el = $(`chip-${kind}-${c}`);
  el.classList.add('on');
  clearTimeout(chipTimers[`${kind}${c}`]);
  chipTimers[`${kind}${c}`] = setTimeout(() => el.classList.remove('on'), 900);
}

function syncSources() {
  $('two').hidden = mode !== 'two';
  $('stereo').hidden = mode !== 'stereo';
  for (let c = 0; c < 2; c++) $(`file-${c}`).hidden = spk[c].src !== 'file';
}

function bind() {
  for (let c = 0; c < 2; c++) {
    $(`src-${c}`).value = spk[c].src;
    $(`src-${c}`).onchange = (e) => {
      spk[c].src = e.target.value;
      if (spk[c].src === 'mic' && spk[1 - c].src === 'mic') { spk[1 - c].src = 'file'; $(`src-${1 - c}`).value = 'file'; }
      syncSources();
    };
    $(`file-${c}`).onchange = (e) => { spk[c].file = e.target.files[0] || null; };
  }
  for (const r of document.querySelectorAll('input[name=mode]')) r.onchange = (e) => { mode = e.target.value; syncSources(); };
  $('stereo-file').onchange = (e) => { stereoFile = e.target.files[0] || null; };
  $('bg-file').onchange = (e) => { bgFile = e.target.files[0] || null; $('bg-clear').hidden = !bgFile; };
  $('bg-clear').onclick = () => { bgFile = null; $('bg-file').value = ''; $('bg-clear').hidden = true; };
  $('bg-gain').oninput = (e) => {
    $('bg-gain-v').textContent = `${Math.round(20 * Math.log10(+e.target.value || 1e-4))} dB`;
    if (graph && graph.bgGain) graph.bgGain.gain.value = +e.target.value;
  };
  for (const k of ['eot', 'int']) $(`th-${k}`).oninput = (e) => { $(`th-${k}-v`).textContent = (+e.target.value).toFixed(2); };
  $('load').onclick = () => loadModel();
  $('start').onclick = start;
  $('stop').onclick = () => stop();
  syncSources();
  $('bg-gain').oninput({ target: $('bg-gain') });
  if (!window.crossOriginIsolated) $('note-iso').hidden = false;
  detectBackend().then((b) => { $('auto-label').textContent = `Auto (${b === 'webgpu' ? 'WebGPU' : 'CPU'})`; });
}

// --- drawing ---------------------------------------------------------------------------------
function draw() {
  requestAnimationFrame(draw);
  const cv = $('timeline');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue('--muted').trim(), line = css.getPropertyValue('--line').trim();
  const lane = (H - 20) / 2, x = (t) => W - (lastT - t + 1) * (W / NF);
  const th = theta();
  g.font = '12px system-ui, sans-serif';
  for (let c = 0; c < 2; c++) {
    const y0 = c * lane, y = (v) => y0 + lane - 6 - v * (lane - 18);
    g.strokeStyle = line;
    g.beginPath(); g.moveTo(0, y0 + lane - 0.5); g.lineTo(W, y0 + lane - 0.5); g.stroke();
    g.fillStyle = fg;
    g.fillText(`Speaker ${spk[c].name}`, 6, y0 + 14);
    if (!hist || lastT < 0) continue;
    const h = hist[c], first = Math.max(0, lastT - NF + 1), dx = W / NF;
    for (let t = first; t <= lastT; t++) {
      const i = t % NF;
      if (h.speak[i] > 0.5) { g.fillStyle = COL.speak; g.fillRect(x(t), y0 + 2, dx + 0.5, lane - 4); }
      const lv = Math.min(1, Math.sqrt(h.level[i]) * 3);
      g.fillStyle = COL.level;
      g.fillRect(x(t), y(0) - lv * (lane - 18) * 0.5, Math.max(1, dx - 1), lv * (lane - 18) * 0.5);
    }
    for (const k of ['eot', 'int']) {
      g.setLineDash([4, 4]); g.strokeStyle = COL[k]; g.globalAlpha = 0.5;
      g.beginPath(); g.moveTo(0, y(th[k])); g.lineTo(W, y(th[k])); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1; g.lineWidth = 1.6;
      g.beginPath();
      for (let t = first; t <= lastT; t++) { const v = h[k][t % NF]; t === first ? g.moveTo(x(t) + dx / 2, y(v)) : g.lineTo(x(t) + dx / 2, y(v)); }
      g.stroke(); g.lineWidth = 1;
    }
    for (const ev of events) {
      if (ev.c !== c) continue;
      const ex = x(ev.t) + dx;
      g.fillStyle = COL[ev.kind];
      g.fillRect(ex - 1, y0 + 2, 2, lane - 4);
      g.beginPath(); g.moveTo(ex - 5, y0 + 18); g.lineTo(ex + 5, y0 + 18); g.lineTo(ex, y0 + 25); g.fill();
      if (ev.re) { g.fillStyle = fg; g.fillText('re', ex + 4, y0 + 30); }
    }
  }
  g.fillStyle = fg;
  for (let s = 0; s <= SPAN; s += 5) {
    const xx = W - s * FPS * (W / NF);
    g.fillText(s === 0 ? 'now' : `-${s} s`, Math.min(W - 28, Math.max(2, xx - 10)), H - 4);
  }
  if (stats) {
    $('st-ms').textContent = stats.calls ? `${stats.ms.toFixed(0)} ms` : '–';
    const per = 80 * framesPerCall;
    $('st-rtf').textContent = stats.calls ? (stats.ms / per).toFixed(2) : '–';
    $('st-lag').textContent = stats.calls ? `${stats.lag.toFixed(0)} ms` : '–';
    $('st-time').textContent = lastT >= 0 ? `${(lastT / FPS).toFixed(1)} s` : '–';
    for (let c = 0; c < 2; c++) $(`cnt-${c}`).textContent = `${stats.counts[c].eot} ends, ${stats.counts[c].int} floor takes`;
  }
}

bind();
draw();
