// Run bench.html in headless Chromium: node run_chromium.mjs [ep] [encoder file] [blocks] [wasm threads] [K]
// Here (no GPU) WebGPU falls back to SwiftShader: fine for correctness, meaningless for speed.
import { chromium } from 'playwright-core';
const [ep = 'wasm', enc = 'encoder_k1_w16.onnx', steps = '100', threads = '4', k = '1'] = process.argv.slice(2);
const b = await chromium.launch({ executablePath: process.env.CHROME || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  headless: true, args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader'] });
const p = await b.newPage();
p.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text().slice(0, 300)); });
await p.goto(`http://localhost:8765/bench.html?ep=${ep}&enc=${enc}&steps=${steps}&threads=${threads}&k=${k}`);
await p.waitForFunction(() => window.__result, null, { timeout: 1800000 });
console.log(JSON.stringify(await p.evaluate(() => window.__result)));
await b.close();
