"""GPU feature caching: frozen causal Cat encoder over 24 kHz stereo audio.

Writes /work/feats/{split}/{cid}.npy: float16 [T, 2, 5888] = taps of top-stage layers
(7, 15, 23, 31) x 1280, then the 768-d final output, on the 80 ms grid. Frame k is
computed only from audio before (k+1)*80 ms.

    modal run encode.py::main --splits oto,tbdev
"""
import json
import time

import modal

from common import WORK, VOLUMES, cpu_image, gpu_image, gpu_monitor, setup_path, work

app = modal.App('ssl-turn-encode')
MODEL_DIR = f'{WORK}/models/cat'


@app.function(image=gpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=3600)
def fetch_cat():
    import os
    setup_path()
    import cat_encoder as ce
    if os.path.exists(f'{MODEL_DIR}/cat_encoder.safetensors'):
        return json.load(open(f'{MODEL_DIR}/cat_encoder.safetensors.json'))
    ce.fetch_code(MODEL_DIR)
    _, ident = ce.fetch_encoder_weights(MODEL_DIR)
    work.commit()
    return ident


@app.function(image=gpu_image, volumes=VOLUMES, gpu='L4', cpu=8, memory=24576, timeout=7200)
def encode(items, batch_waves=32, chunk_frames=50, dtype='fp32'):
    import os, threading
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    setup_path()
    import cat_encoder as ce
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[dtype]
    enc = ce.build(MODEL_DIR, f'{MODEL_DIR}/cat_encoder.safetensors', device='cuda', dtype=run_dtype)

    def load(item):
        split, cid = item
        a = np.load(f'{WORK}/audio/{split}/{cid}.npy')
        return item, a

    stats, stop = [], threading.Event()
    threading.Thread(target=gpu_monitor, args=(stats, stop), daemon=True).start()
    with ThreadPoolExecutor(8) as pool:
        loaded = list(pool.map(load, items))
    loaded.sort(key=lambda x: -len(x[1]))
    # Precision sanity on 20 s of the first conversation: run dtype (TF32 matmuls when fp32)
    # against strict fp32 with TF32 off.
    first = torch.from_numpy(loaded[0][1][:20 * ce.SAMPLE_RATE].T.astype(np.float32)).cuda()
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = ce.build(MODEL_DIR, f'{MODEL_DIR}/cat_encoder.safetensors', device='cuda').stream(first, 25)['final']
        torch.backends.cuda.matmul.allow_tf32 = True
        low = enc.stream(first.to(run_dtype), 25)['final'].float()
    cos = torch.nn.functional.cosine_similarity(ref, low, dim=-1)
    print(f'{dtype}(tf32) vs strict fp32 cosine: min {cos.min():.4f} mean {cos.mean():.4f}', flush=True)
    del ref
    torch.cuda.empty_cache()

    waves = [(item, ch, a[:, ch].copy()) for item, a in loaded for ch in (0, 1)]
    del loaded
    results, saved = {}, 0
    t0, audio_s = time.time(), 0.0
    for b in range(0, len(waves), batch_waves):
        group = waves[b:b + batch_waves]
        n = max(len(w) for _, _, w in group)
        n = (n + ce.HOP - 1) // ce.HOP * ce.HOP
        x = np.zeros((len(group), n), np.float32)
        for i, (_, _, w) in enumerate(group):
            x[i, :len(w)] = w
        with torch.no_grad():
            out = enc.stream(torch.from_numpy(x).cuda().to(run_dtype), chunk_frames, out_device='cpu',
                             out_dtype=torch.float16)
        feats = torch.cat([out['taps'].flatten(2), out['final']], -1).numpy()
        for i, (item, ch, w) in enumerate(group):
            T = len(w) // ce.HOP
            results.setdefault(item, {})[ch] = feats[i, :T].copy()  # copy: free the batch buffer
            audio_s += len(w) / ce.SAMPLE_RATE
            if len(results[item]) == 2:  # both channels done: save and release (bounded RAM)
                chans = results.pop(item)
                split, cid = item
                T = min(len(chans[0]), len(chans[1]))
                os.makedirs(f'{WORK}/feats/{split}', exist_ok=True)
                np.save(f'{WORK}/feats/{split}/{cid}.npy', np.stack([chans[0][:T], chans[1][:T]], 1))
                saved += 1
                if saved % 10 == 0:
                    work.commit()
        for j in range(b, b + len(group)):  # release source audio once encoded
            waves[j] = None
        print(f'batch {b // batch_waves}: {len(group)} waves, {audio_s / (time.time() - t0):.0f} channel-s/s', flush=True)
    work.commit()
    stop.set()
    util = [u for u, _ in stats]
    mem = [m for _, m in stats]
    return dict(items=len(items), channel_seconds=audio_s, wall_s=time.time() - t0,
                gpu_util_mean=float(np.mean(util)) if util else None, gpu_mem_max_mib=max(mem) if mem else None)


@app.function(image=cpu_image, volumes=VOLUMES, timeout=600)
def todo(splits):
    import os
    out = []
    for split in splits:
        have = {f[:-4] for f in os.listdir(f'{WORK}/feats/{split}')} if os.path.isdir(f'{WORK}/feats/{split}') else set()
        for f in sorted(os.listdir(f'{WORK}/audio/{split}')):
            if f.endswith('.npy') and f[:-4] not in have:
                out.append((split, f[:-4]))
    return out


@app.local_entrypoint()
def main(splits: str = 'oto,tbdev', batch_waves: int = 32, dtype: str = 'fp32'):
    print('weights', fetch_cat.remote())
    items = todo.remote(splits.split(','))
    print('to encode', len(items))
    if items:
        print(encode.remote(items, batch_waves, 50, dtype))
