"""bench.html in headless Chromium with WebGPU on a real NVIDIA GPU (Modal L4, Vulkan).

    BENCH_DIR=/path/to/bundle modal run modal_webgpu.py --runs 'ep=webgpu&enc=encoder_k2_w16.onnx&k=2&steps=300,...'

BENCH_DIR holds bench.html, serve.py, turn_stream.js, sample.f32, ref.json and
node_modules/onnxruntime-web/dist; the encoder/head files live on the `turn-web` volume under
models/ (modal volume put turn-web web_models/X.onnx models/X.onnx).

Gotchas: Modal mounts NVIDIA's Vulkan driver, but its ICD json points at libGLX_nvidia, which
fails headless ("Could not get vkCreateInstance"); pointing the ICD at libEGL_nvidia works.
Chrome 131 on this driver exposes no shader-f16, so *_fp16 encoders fall back to CPU kernels.
"""
import os

import modal

app = modal.App('turn-webgpu-bench')
img = (modal.Image.debian_slim(python_version='3.11')
       .apt_install('libvulkan1', 'libxext6', 'libx11-6', 'libx11-xcb1', 'libxcb1', 'vulkan-tools')
       .pip_install('playwright==1.49.1')
       .run_commands('playwright install --with-deps chromium')
       .env({'NVIDIA_DRIVER_CAPABILITIES': 'all'})
       .add_local_dir(os.environ.get('BENCH_DIR', '.'), '/web'))
vol = modal.Volume.from_name('turn-web')
FLAGS = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-vulkan=native',
         '--use-angle=vulkan', '--disable-vulkan-surface', '--enable-gpu', '--no-sandbox']


@app.function(image=img, gpu='L4', volumes={'/vol': vol}, timeout=1800, cpu=4, memory=16384)
def bench(runs):
    import json
    import subprocess
    import time
    os.symlink('/vol/models', '/web/models')
    icd = json.load(open('/etc/vulkan/icd.d/nvidia_icd.json'))
    icd['ICD']['library_path'] = 'libEGL_nvidia.so.0'
    json.dump(icd, open('/tmp/nv_egl.json', 'w'))
    os.environ['VK_ICD_FILENAMES'] = '/tmp/nv_egl.json'
    srv = subprocess.Popen(['python', 'serve.py'], cwd='/web')
    time.sleep(1)
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chromium', headless=True, args=FLAGS)
        pg = b.new_page()
        pg.goto('http://localhost:8765/serve.py')
        out['adapter'] = pg.evaluate('''async () => { if (!navigator.gpu) return 'no navigator.gpu';
            const a = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
            if (!a) return 'null adapter';
            return {vendor: a.info.vendor, arch: a.info.architecture, f16: a.features.has('shader-f16')}; }''')
        print('adapter', out['adapter'], flush=True)
        out['results'] = []
        for q in runs:
            pg.goto(f'http://localhost:8765/bench.html?{q}')
            pg.wait_for_function('() => window.__result', timeout=1500000)
            r = pg.evaluate('() => window.__result')
            print(json.dumps(r), flush=True)
            out['results'].append(r)
        b.close()
    srv.kill()
    return out


@app.local_entrypoint()
def main(runs: str = 'ep=webgpu&enc=encoder_k2_w16.onnx&k=2&steps=300'):
    bench.remote(runs.split(','))
