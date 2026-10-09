"""Background-speech robustness of the ssl_turn floor model on TurnBench dev.

Same mixtures as bgspeech/ (VAP): a far-field "podcast playing" (another TB dev
conversation, both speakers) is added to one channel ("user") of each evaluated
conversation at a given SNR. Only that channel is re-encoded with Cat; the clean
channel's cached features are reused. Inference runs in the same GPU container
(no mixed features are stored), writing /work/runs/<out_run>/probs.npz with keys
'<model>@<cond>/tbdev/<cid>/{post,silent,fine}' for score.py-style scoring.

    modal run bgmix.py --plan plan.json --masks masks.npz --run r012_fine \
        --models fine1_bal1_s1,fine1_bal1_s2 --out-run bg_r012
"""
import json
import time
from pathlib import Path

import modal

from common import VOLUMES, gpu_image, setup_path, work

image = gpu_image
if modal.is_local():  # in the container this module lives at /root/bgmix.py
    image = gpu_image.add_local_file(str(Path(__file__).resolve().parents[2] / 'bgspeech' / 'mixing.py'),
                                     '/root/bgspeech/mixing.py')
app = modal.App('ssl-turn-bgmix')
CONDITIONS = {'snr10': (10.0, False), 'snr5': (5.0, False), 'snr0': (0.0, False),
              'gate5': (5.0, True), 'gate0': (0.0, True)}


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=7200)
def encode_infer(plan, masks, run, names, out_run, conds, batch_waves=16):
    import hashlib
    import os
    import sys
    import numpy as np
    import torch
    setup_path()
    sys.path.insert(0, '/root/bgspeech')
    sys.path.insert(0, '/root/ssl_turn/pipeline')
    import cat_encoder as ce
    import mixing
    import train as tr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sr = ce.SAMPLE_RATE
    ids = plan['subset']

    # 1. Mixed user channels (24 kHz, same recipe as the VAP run; rng seeded per conversation).
    t0 = time.time()
    audio = {c: np.load(f'/work/audio/tbdev/{c}.npy').astype(np.float32) for c in ids}
    donors = {}
    for c in ids:
        for d in plan['items'][c]['donors']:
            if d not in donors:
                a = np.load(f'/work/audio/tbdev/{d}.npy').astype(np.float32)
                donors[d] = a[:, 0] + a[:, 1]
    waves = []
    for c in ids:
        user = plan['items'][c]['user']
        x = audio[c][:, user - 1]
        rng = np.random.default_rng(int(hashlib.sha256(c.encode()).hexdigest()[:8], 16))
        bg = mixing.background_track([donors[d] for d in plan['items'][c]['donors']], len(x), sr, rng)
        mask = masks[c]
        for cond in conds:
            snr, gated = CONDITIONS[cond]
            b = bg * mixing.smooth_gate(mask, 100.0, len(x), sr) if gated else bg
            waves.append(((c, cond), mixing.mix(x, sr, b, snr, mask, bg_level=bg)))
    del donors
    print(f'mixed {len(waves)} channels in {time.time() - t0:.0f}s', flush=True)

    # 2. Cat-encode only the mixed channels.
    enc = ce.build('/work/models/cat', '/work/models/cat/cat_encoder.safetensors', device='cuda')
    cols = tr.columns([15, 23, 31])
    tr.LOADED = [15, 23, 31]
    feats = {}
    waves.sort(key=lambda w: -len(w[1]))
    t0, done = time.time(), 0.0
    for b in range(0, len(waves), batch_waves):
        group = waves[b:b + batch_waves]
        n = (max(len(w) for _, w in group) + ce.HOP - 1) // ce.HOP * ce.HOP
        x = np.zeros((len(group), n), np.float32)
        for i, (_, w) in enumerate(group):
            x[i, :len(w)] = w
        with torch.no_grad():
            out = enc.stream(torch.from_numpy(x).cuda(), 50, out_device='cpu', out_dtype=torch.float16)
        f = torch.cat([out['taps'].flatten(2), out['final']], -1).numpy()
        for i, (key, w) in enumerate(group):
            feats[key] = f[i, :len(w) // ce.HOP][:, cols].copy()
            done += len(w) / sr
        print(f'batch {b // batch_waves}: {done / (time.time() - t0):.0f} channel-s/s', flush=True)
    del enc, waves
    torch.cuda.empty_cache()

    # 3. Inference with the saved checkpoints, clean (cached features) and each condition.
    models, configs = {}, {}
    for nme in names:
        ck = torch.load(f'/work/runs/{run}/{nme}.pt', map_location='cuda')
        configs[nme] = ck['cfg']
        models[nme] = tr.build_model(ck['cfg']).cuda()
        models[nme].load_state_dict(ck['state'])
    clean = {c: np.ascontiguousarray(np.load(f'/work/feats/tbdev/{c}.npy', mmap_mode='r')[..., cols]) for c in ids}
    probs = {}
    for cond in ['clean'] + list(conds):
        Xs, offs = [], [0]
        for c in ids:
            x = clean[c]
            if cond != 'clean':
                m = feats[(c, cond)]
                T = min(len(x), len(m))
                x = x[:T].copy()
                x[:, plan['items'][c]['user'] - 1] = m[:T]
            Xs.append(x)
            offs.append(offs[-1] + len(x))
        X = torch.from_numpy(np.concatenate(Xs)).cuda()
        out = tr.infer_all(models, configs, (('tbdev', X, offs, ids),), 'cuda')
        for k, v in out.items():
            mdl, rest = k.split('/', 1)
            probs[f'{mdl}@{cond}/{rest}'] = v
        del X
    os.makedirs(f'/work/runs/{out_run}', exist_ok=True)
    np.savez_compressed(f'/work/runs/{out_run}/probs.npz', **probs)
    json.dump(dict(plan=plan, run=run, models=names, conds=conds), open(f'/work/runs/{out_run}/bgmix.json', 'w'))
    work.commit()
    return len(probs)


@app.local_entrypoint()
def main(plan: str, masks: str, run: str = 'r012_fine', models: str = 'fine1_bal1_s1,fine1_bal1_s2',
         out_run: str = 'bg_r012', conds: str = 'snr10,snr5,snr0,gate5,gate0'):
    import numpy as np
    p = json.load(open(plan))
    with np.load(masks) as z:
        m = {k: z[k] for k in z.files}
    print(encode_infer.remote(p, m, run, models.split(','), out_run, conds.split(',')))
