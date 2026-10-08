"""GPU feature caching with MOSS-Transcribe-Diarize's encoder, exactly causal (trailing windows).

Every `step` frames (default 2, i.e. 160 ms), each channel's last <= 30 s of 16 kHz
audio ending at the step is zero-padded to 30 s like the upstream processor's final
chunk, Whisper log-mel is computed on the GPU, and the encoder's last 12.5 Hz token is
kept: mean over its four 50 Hz frames of hidden states 8/16/24 plus the final
layer-normed output (4 x 1024). Between steps the latest available feature is held,
so frame k only uses audio before (k+1)*80 ms.

Audio is the cached 24 kHz causal resample (prep.py), decimated causally to 16 kHz here.
Output: /work/feats_mtd/{split}/{cid}.npy float16 [T, 2, 4096].
"""
import json
import time

import modal

from common import VOLUMES, gpu_image, gpu_monitor, setup_path, work

app = modal.App('ssl-turn-encode-mtd')
MODEL_DIR = '/work/models/mtd'
LAYERS = (8, 16, 24)


@app.function(image=gpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=1800)
def fetch_mtd():
    import os
    setup_path()
    import mtd_encoder as me
    if not os.path.exists(f'{MODEL_DIR}/mtd_encoder.safetensors'):
        me.fetch(MODEL_DIR)
        work.commit()
    return json.load(open(f'{MODEL_DIR}/mtd_encoder.safetensors.json'))


def to16k(x24):
    """24 -> 16 kHz with the same one-sided FIR design as prep.causal_resample (no lookahead)."""
    import numpy as np
    from scipy.signal import firwin, upfirdn
    up, down = 2, 3
    half = 10 * max(up, down)
    taps = firwin(2 * half + 1, 1 / max(up, down), window=('kaiser', 5.0)) * up
    count = (len(x24) * up + down - 1) // down
    return upfirdn(taps, x24.astype(np.float32), up=up, down=down, axis=0)[:count].astype(np.float32)


@app.function(image=gpu_image, volumes=VOLUMES, gpu='H100', cpu=8, memory=32768, timeout=7200)
def encode(items, step=2, batch=96):
    import os, threading
    import numpy as np
    import torch
    setup_path()
    import mtd_encoder as me
    torch.backends.cuda.matmul.allow_tf32 = True
    model = me.build(MODEL_DIR, f'{MODEL_DIR}/mtd_encoder.safetensors', taps=LAYERS, device='cuda',
                     dtype=torch.bfloat16)
    enc, fe = model.encoder, model.fe
    stats, stop = [], threading.Event()
    threading.Thread(target=gpu_monitor, args=(stats, stop), daemon=True).start()
    W = me.CHUNK_SAMPLES
    hann = torch.hann_window(fe.n_fft, device='cuda')
    mel_filters = torch.from_numpy(fe.mel_filters).to('cuda', torch.float32)

    def logmel(x):
        """Same ops as WhisperFeatureExtractor._torch_extract_fbank_features (batched branch),
        kept on the GPU: per-window max-8 clamp, so each window is normalized on its own."""
        stft = torch.stft(x, fe.n_fft, fe.hop_length, window=hann, return_complex=True)
        mag = (stft[..., :-1].abs() ** 2).contiguous()
        log_spec = torch.clamp(mel_filters.T @ mag, min=1e-10).log10()
        peak = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        return (torch.maximum(log_spec, peak - 8.0) + 4.0) / 4.0
    t0, done_s = time.time(), 0.0
    for split, cid in items:
        a24 = np.load(f'/work/audio/{split}/{cid}.npy').astype(np.float32)
        chans = [to16k(a24[:, c]) for c in (0, 1)]
        n = min(len(c) for c in chans)
        T = n // me.TOKEN_SAMPLES                      # 80 ms frames
        ks = np.arange(step - 1, T, step)              # frames whose end gets a fresh encode
        out = np.zeros((T, 2, 4 * me.DIM), np.float16)
        for c in (0, 1):
            wav = torch.from_numpy(chans[c][:n]).cuda()
            feats = []
            for b in range(0, len(ks), batch):
                ends = (ks[b:b + batch] + 1) * me.TOKEN_SAMPLES
                win = torch.zeros((len(ends), W), device='cuda')
                lens = np.minimum(ends, W)
                for i, (e, L) in enumerate(zip(ends, lens)):
                    win[i, :L] = wav[e - L:e]
                mel = logmel(win).to(torch.bfloat16)  # per-window Whisper log-mel, on GPU
                with torch.no_grad():
                    o = enc(mel, output_hidden_states=True)
                tok = torch.as_tensor(lens // me.TOKEN_SAMPLES, device='cuda')
                frames = (tok[:, None] - 1) * me.MERGE + torch.arange(me.MERGE, device='cuda')
                rows = torch.arange(len(ends), device='cuda')[:, None]
                parts = [o.hidden_states[l][rows, frames].float().mean(1) for l in LAYERS]
                parts.append(o.last_hidden_state[rows, frames].float().mean(1))
                feats.append(torch.cat(parts, -1).half().cpu())
            f = torch.cat(feats).numpy()
            # Hold the latest fresh feature until the next one (frame k uses ks <= k).
            idx = np.searchsorted(ks, np.arange(T), side='right') - 1
            valid = idx >= 0
            out[valid, c] = f[idx[valid]]
        os.makedirs(f'/work/feats_mtd/{split}', exist_ok=True)
        np.save(f'/work/feats_mtd/{split}/{cid}.tmp.npy', out)
        os.replace(f'/work/feats_mtd/{split}/{cid}.tmp.npy', f'/work/feats_mtd/{split}/{cid}.npy')  # atomic
        done_s += 2 * n / me.SAMPLE_RATE
        util = [u for u, _ in stats[-20:]]
        print(f'{split}/{cid}: {T} frames; {done_s / (time.time() - t0):.1f} channel-s/s; '
              f'GPU {np.mean(util) if util else -1:.0f}%', flush=True)
        if len(done_items := locals().setdefault('_done', [])) % 5 == 4:
            work.commit()
        done_items.append(cid)
    work.commit()
    stop.set()
    util = [u for u, _ in stats]
    return dict(items=len(items), channel_seconds=done_s, wall_s=time.time() - t0,
                gpu_util_mean=float(np.mean(util)) if util else None, gpu_mem_max_mib=max(m for _, m in stats))


@app.local_entrypoint()
def main(n_train: int = 32, step: int = 2, splits: str = 'oto,tbdev'):
    import io
    print(fetch_mtd.remote())
    buf = io.BytesIO()
    for chunk in work.read_file('split.json'):
        buf.write(chunk)
    split = json.loads(buf.getvalue())['splits']
    tb = sorted(e.path.split('/')[-1][:-4] for e in work.listdir('feats/tbdev'))
    dirs = {x.path.split('/')[-1] for x in work.listdir('feats_mtd')}
    have = {sp: ({e.path.split('/')[-1][:-4] for e in work.listdir(f'feats_mtd/{sp}')} if sp in dirs else set())
            for sp in ('oto', 'tbdev')}  # per split: conversation ids overlap across datasets
    items = []
    if 'oto' in splits:
        items += [('oto', c) for c in split['train'][:n_train] + split['dev'] if c not in have['oto']]
    if 'tbdev' in splits:
        items += [('tbdev', c) for c in tb if c not in have['tbdev']]
    print('items', len(items))
    print(encode.remote(items, step))
