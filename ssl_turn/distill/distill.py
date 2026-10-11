"""Distill the deployed pipeline's model part (streaming FastConformer + floor head) into one small
causal network that reads 16 kHz audio (student.py). The commit policy (policy.py) and Silero VAD
stay as they are; the student replaces the 114M encoder and the head.

    cd ssl_turn/distill
    modal run distill.py::prep                                  # 16 kHz audio (+ bg-augmented copies)
    modal run distill.py::teacher --run r019_asr_bgaug --name bgaug_s2 --tag r019s2
    modal run distill.py::train --run d001 --configs configs/d001.json --teacher r019s2
    modal run distill.py::eot --run d001 --names s8m,s3m --out tracks.npz   # LiveKit eot-bench en

Volume layout (ssl-turn-work):
  audio16/{oto,tbdev}/{cid}.npy    float16 [T * 1280, 2]: the 16 kHz audio the teacher's
                                   FastConformer features were computed from (encode_asr.to16k)
  audio16_aug/oto/{cid}.npy        the same with one background per channel, recipe of
                                   bgmix.encode_aug_asr (far .5 / tv .2 / babble .2 / music .1)
  teacher/{tag}/{split}/{cid}.npz  the teacher head's logits on the cached features (exact causal
                                   chunked inference): floor, future, act, fine, vap, yld
  runs/{run}/                      student checkpoints, train.json, probs.npz (score.py format:
                                   '{name}/{split}/{cid}/{post,silent,fine,yld}')

Training: random 30 s crops of the 131 otoSpeech train conversations. Per frame the student
matches the teacher's distributions (soft cross-entropy on floor, projections, acts, fine labels,
VAP; BCE on yld), the gold labels (model.loss, weight `gt`), and the teacher's 1024-d encoder frame
(standardized, MSE, weight `feat`). With probability `bg_aug` a crop takes one channel from the
background-augmented copy while every target stays the clean teacher's (clean teacher, noisy
student).
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pipeline'))
from common import VOLUMES, WORK, _code, gpu_monitor, setup_path, work  # noqa: E402

BG = HERE.parent.parent / 'bgspeech'
image = _code(modal.Image.debian_slim(python_version='3.12')
              .pip_install('torch==2.8.0', 'transformers==5.19.0', 'safetensors', 'numpy', 'scipy', 'soundfile',
                           'huggingface_hub', 'httpx', 'pyarrow==19.0.1'))
if modal.is_local():
    image = (image.add_local_file(str(BG / 'mixing.py'), '/root/bgspeech/mixing.py')
             .add_local_file(str(BG / 'bench.py'), '/root/bgspeech/bench.py'))
app = modal.App('ssl-turn-distill')
FRAME = 1280
EOT_DATASET = 'livekit/eot-bench-data'
EOT_REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'


def _paths():
    setup_path()
    for p in ('/root/ssl_turn/distill', '/root/ssl_turn/pipeline', '/root/bgspeech'):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---- 1. 16 kHz audio, clean and background-augmented ---------------------------------------------

ASR_AUG_STYLES = {'far': 0.5, 'tv': 0.2, 'babble': 0.2, 'music': 0.1}  # bgmix.ASR_AUG_STYLES


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=16384, timeout=7200)
def prep_group(items, donor_ids, snr_range=(-5.0, 20.0)):
    """items: [(split, cid, aug)]. Same resampling as encode_asr (24 kHz cache -> to16k), trimmed
    to the teacher's T = n // 1280 frames."""
    import hashlib
    import numpy as np
    import soundfile as sf
    _paths()
    import bench
    import encode_asr as ea
    import mixing
    music = None
    for split, cid, aug in items:
        a = np.load(f'{WORK}/audio/{split}/{cid}.npy').astype(np.float32)
        T = np.load(f'{WORK}/feats_asr/{split}/{cid}.npy', mmap_mode='r').shape[0]
        out = f'{WORK}/audio16/{split}/{cid}.npy'
        if not os.path.exists(out):
            w = np.stack([ea.to16k(a[:, c]) for c in (0, 1)], 1)[:T * FRAME]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out + '.tmp.npy', w.astype(np.float16))
            os.replace(out + '.tmp.npy', out)
        if not aug or os.path.exists(f'{WORK}/audio16_aug/{split}/{cid}.npy'):
            continue
        if music is None:
            mdir = f'{WORK}/musan/fma_train'
            music = []
            for f in sorted(os.listdir(mdir)):
                x, s = sf.read(f'{mdir}/{f}', dtype='float32', always_2d=True)
                music.append(mixing.resample_to(x.mean(1), s, bench.SR))
        act = np.load(f'{WORK}/labels/oto/{cid}.npz')['activity']
        rng = np.random.default_rng(int(hashlib.sha256(f'aug-asr:{cid}'.encode()).hexdigest()[:8], 16))
        styles, probs = list(ASR_AUG_STYLES), np.array(list(ASR_AUG_STYLES.values()))
        chans = []
        for c in (0, 1):
            style = styles[int(rng.choice(len(styles), p=probs))]
            picks = rng.choice([d for d in donor_ids if d != cid], 4, replace=False)
            dons = [np.load(f'{WORK}/audio/oto/{d}.npy').astype(np.float32) for d in picks]
            bg = bench.backgrounds(f'train:{cid}:{c}', len(a), dons, music, styles=(style,))[style]
            mask = np.repeat(act[:, c] > 0.5, 8)
            snr = float(rng.uniform(*snr_range))
            chans.append(ea.to16k(mixing.mix(a[:, c], bench.SR, bg, snr, mask))[:T * FRAME])
        n = min(len(x) for x in chans)
        out = f'{WORK}/audio16_aug/{split}/{cid}.npy'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out + '.tmp.npy', np.stack([x[:n] for x in chans], 1).astype(np.float16))
        os.replace(out + '.tmp.npy', out)
        work.commit()
        print(cid, flush=True)
    work.commit()
    return len(items)


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=600)
def _plan():
    work.reload()
    split = json.load(open(f'{WORK}/split.json'))['splits']
    have = sorted(f[:-4] for f in os.listdir(f'{WORK}/feats_asr/oto') if not f.endswith('.tmp.npy'))
    tb = sorted(f[:-4] for f in os.listdir(f'{WORK}/feats_asr/tbdev') if not f.endswith('.tmp.npy'))
    donors = [c for c in have if c not in set(split['train']) | set(split['dev'])][:24]  # bgmix.aug_asr_plan
    train = set(split['train'])
    items = [('oto', c, c in train) for c in have] + [('tbdev', c, False) for c in tb]
    return items, donors


@app.local_entrypoint()
def prep(groups: int = 24):
    items, donors = _plan.remote()
    gs = [items[i::groups] for i in range(groups)]
    print(sum(prep_group.starmap([(g, donors) for g in gs if g])), 'conversations')


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=16384, timeout=3600)
def prep_eotb_lang(lang):
    """16 kHz audio of latency/eotb_data.py's sessions (same rows, order and frame counts; agent
    channel zeros), so students can train on the voice-agent turns too."""
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    session_frames = 3000  # eotb_data.SESSION_FRAMES
    path = hf_hub_download(EOT_DATASET, f'data/{lang}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=EOT_REVISION)
    rows = pq.read_table(path, columns=['id', 'audio', 'silence_spans']).to_pylist()
    os.makedirs(f'{WORK}/audio16/eotb', exist_ok=True)
    chunks, k = [], 0

    def flush():
        nonlocal chunks, k
        if chunks:
            name = f'{lang}_{k:02d}'
            T = np.load(f'{WORK}/feats_asr/eotb/{name}.npy', mmap_mode='r').shape[0]
            w = np.concatenate(chunks)
            assert len(w) == T * FRAME, (name, len(w), T)
            np.save(f'{WORK}/audio16/eotb/{name}.npy', w.astype(np.float16))
            chunks, k = [], k + 1

    for r in rows:
        wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // FRAME
        spans = [x for x in r['silence_spans'] if x['end'] - x['start'] > 0]
        if T < 10 or not spans:
            continue
        x = np.zeros((T * FRAME, 2), np.float32)
        x[:, 0] = wav[:T * FRAME]
        chunks.append(x)
        if sum(len(c) for c in chunks) // FRAME >= session_frames:
            flush()
    flush()
    work.commit()
    return lang, k


@app.local_entrypoint()
def prep_eotb(langs: str = 'de,fr,it,nl,pt,ar,hi,id,ko,tr'):
    for r in prep_eotb_lang.map(langs.split(',')):
        print(r)


# ---- 2. teacher logits ---------------------------------------------------------------------------

def _head(ck):
    _paths()
    import train as tr
    return tr.build_model(ck['cfg'])


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=32768, timeout=7200)
def teacher_logits(run, name, tag, splits=('oto', 'tbdev')):
    """Exact chunked causal inference of one teacher head over cached FastConformer features."""
    import numpy as np
    import torch
    _paths()
    import model as m
    import train as tr
    ck = torch.load(f'{WORK}/runs/{run}/{name}.pt', map_location='cuda', weights_only=False)
    net = _head(ck).cuda().eval()
    net.load_state_dict(ck['state'])
    ctx = tr.model_context(ck['cfg'])
    n = 0
    for split in splits:
        d = f'{WORK}/feats_asr/{split}'
        os.makedirs(f'{WORK}/teacher/{tag}/{split}', exist_ok=True)
        for f in sorted(os.listdir(d)):
            if f.endswith('.tmp.npy'):
                continue
            cid = f[:-4]
            X = torch.from_numpy(np.load(f'{d}/{f}')).cuda()
            T, outs = len(X), []
            with torch.no_grad():
                for s in range(0, T, 1000):
                    lo = max(0, s - ctx)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = net(None, X[lo:min(T, s + 1000)][None])
                    outs.append({k: v[0, s - lo:].float() for k, v in o.items()
                                 if k in ('floor', 'future', 'act', 'fine', 'vap', 'yld')})
            res = {k: torch.cat([o[k] for o in outs]).cpu().numpy().astype(np.float16) for k in outs[0]}
            np.savez(f'{WORK}/teacher/{tag}/{split}/{cid}.npz', **res)
            n += 1
        work.commit()
    json.dump(dict(run=run, name=name, cfg=ck['cfg'], vap_bins=list(m.VAP_BINS)),
              open(f'{WORK}/teacher/{tag}/meta.json', 'w'))
    work.commit()
    return n


@app.local_entrypoint()
def teacher(run: str = 'r019_asr_bgaug', name: str = 'bgaug_s2', tag: str = 'r019s2', splits: str = 'oto,tbdev'):
    print(teacher_logits.remote(run, name, tag, tuple(splits.split(','))), 'conversations')


# ---- 3. student training --------------------------------------------------------------------------

def _load(split, cids, tag, device, labels=True, aug=False, feats=True):
    """Concatenated per-frame arrays on `device`: audio A [N * 1280, 2] (and the augmented copy),
    teacher logits, gold labels (+ pause targets), teacher features [N, 2, 1024]."""
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    _paths()
    import labels as lb

    def length(cid):
        T = np.load(f'{WORK}/audio16/{split}/{cid}.npy', mmap_mode='r').shape[0] // FRAME
        with np.load(f'{WORK}/teacher/{tag}/{split}/{cid}.npz') as z:
            T = min(T, len(z['floor']))
        if aug and os.path.exists(f'{WORK}/audio16_aug/{split}/{cid}.npy'):  # extra conversations have none
            T = min(T, np.load(f'{WORK}/audio16_aug/{split}/{cid}.npy', mmap_mode='r').shape[0] // FRAME)
        if labels:
            with np.load(f'{WORK}/labels/{split}/{cid}.npz') as z:
                T = min(T, len(z['floor']))
        return T

    def read(cid, T):
        r = dict(a=np.load(f'{WORK}/audio16/{split}/{cid}.npy', mmap_mode='r')[:T * FRAME])
        if aug:
            p = f'{WORK}/audio16_aug/{split}/{cid}.npy'
            r['aug'] = np.load(p, mmap_mode='r')[:T * FRAME] if os.path.exists(p) else r['a']
        if feats:
            r['feat'] = np.load(f'{WORK}/feats_asr/{split}/{cid}.npy', mmap_mode='r')[:T]
        with np.load(f'{WORK}/teacher/{tag}/{split}/{cid}.npz') as z:
            r['t'] = {k: z[k][:T] for k in z.files}
        if labels:
            with np.load(f'{WORK}/labels/{split}/{cid}.npz') as z:
                lab = {k: z[k][:T] for k in ('floor', 'floor_w', 'act', 'act_w', 'future', 'future_w', 'activity')}
                lab['fine'], lab['fine_w'] = z['fine'][:T], (z['fine_w'] if 'fine_w' in z.files else z['act_w'])[:T]
                lab['vap_mask'] = z['vap_mask'][:T] if 'vap_mask' in z.files else np.ones(T, bool)
            lab.update(lb.pause_targets(lab['floor'], lab['floor_w'], lab['act']))
            r['y'] = lab
        return r

    with ThreadPoolExecutor(16) as pool:
        lengths = list(pool.map(length, cids))
        off = np.concatenate([[0], np.cumsum(lengths)]).tolist()
        N = off[-1]
        A = torch.empty((N * FRAME, 2), dtype=torch.float16, device=device)
        Aa = torch.empty((N * FRAME, 2), dtype=torch.float16, device=device) if aug else None
        Fz = torch.empty((N, 2, 1024), dtype=torch.float16, device=device) if feats else None
        Tch, Y = {}, []
        for a, r in zip(off[:-1], pool.map(read, cids, lengths)):
            T = len(r['a']) // FRAME
            A[a * FRAME:(a + T) * FRAME] = torch.from_numpy(np.ascontiguousarray(r['a'])).to(device)
            if aug:
                Aa[a * FRAME:(a + T) * FRAME] = torch.from_numpy(np.ascontiguousarray(r['aug'])).to(device)
            if feats:
                Fz[a:a + T] = torch.from_numpy(np.ascontiguousarray(r['feat'])).to(device)
            for k, v in r['t'].items():
                if k not in Tch:
                    Tch[k] = torch.empty((N,) + v.shape[1:], dtype=torch.float16, device=device)
                Tch[k][a:a + T] = torch.from_numpy(v).to(device)
            if labels:
                Y.append(r['y'])
    Yt = None
    if labels:
        import train as tr
        Yt = tr.stack_labels(Y, device)
    return dict(A=A, Aa=Aa, F=Fz, T=Tch, Y=Yt, off=off)


def soft_ce(s, t, w=None):
    """Soft cross-entropy of student logits s against teacher logits t (last dim), mean or w-weighted."""
    import torch
    p = t.float().softmax(-1)
    ce = -(p * s.float().log_softmax(-1)).sum(-1)
    if w is None:
        return ce.mean()
    w = w.expand_as(ce)
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


KD = dict(floor=1.0, future=1.0, act=0.5, fine=0.5, vap=0.5, yld=1.0)


def kd_loss(out, tb, weights=None):
    import torch.nn.functional as F
    w = {**KD, **(weights or {})}
    parts = {}
    for k in ('floor', 'future', 'act', 'fine', 'vap'):
        if k in out and k in tb and w.get(k, 0) > 0:
            parts[k] = soft_ce(out[k], tb[k])
    if 'yld' in out and 'yld' in tb and w.get('yld', 0) > 0:
        parts['yld'] = F.binary_cross_entropy_with_logits(out['yld'].float(), tb['yld'].float().sigmoid())
    return sum(w[k] * v for k, v in parts.items()), parts


@app.function(image=image, volumes=VOLUMES, gpu='H100', cpu=8, memory=65536, timeout=4 * 3600)
def train_students(run, configs, tag, steps=6000, batch=48, crop=375, eval_every=500, lr=1e-3, seed=0,
                   infer=True, extra_ids=()):
    import threading
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import numpy as np
    import torch
    _paths()
    import model as m
    import student as st
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    work.reload()
    split = json.load(open(f'{WORK}/split.json'))['splits']
    have = {f[:-4] for f in os.listdir(f'{WORK}/teacher/{tag}/oto')}
    train_ids = [c for c in split['train'] if c in have] + [c for c in extra_ids if c in have]
    dev_ids = [c for c in split['dev'] if c in have]
    bg_p = max(c.get('bg_aug', 0.0) for c in configs.values())
    eotb_p = max(c.get('eotb_aug', 0.0) for c in configs.values())
    assert len({c.get('eotb_aug', 0.0) for c in configs.values()}) == 1, 'students in one run share batches'
    t0 = time.time()
    D = _load('oto', train_ids, tag, dev, aug=bg_p > 0)
    Dd = _load('oto', dev_ids, tag, dev)
    De = None
    if eotb_p > 0:  # voice-agent user turns (latency/eotb_data.py sessions), silent agent channel
        eids = sorted(f[:-4] for f in os.listdir(f'{WORK}/teacher/{tag}/eotb'))
        De = _load('eotb', eids, tag, dev)
        estarts = torch.cat([torch.arange(a, b - crop, device=dev) for a, b in zip(De['off'][:-1], De['off'][1:])
                             if b - a > crop])
        print(f'eot-bench sessions: {len(eids)}, {De["off"][-1]} frames', flush=True)
    print(f'loaded {len(train_ids)} train ({D["off"][-1]} frames) / {len(dev_ids)} dev in {time.time() - t0:.0f}s; '
          f'VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)
    # standardization of the teacher's encoder frames (feature distillation target)
    sample = D['F'][torch.randint(len(D['F']), (20000,), device=dev)].float().flatten(0, 1)
    f_mu, f_sd = sample.mean(0), sample.std(0).clamp_min(1e-3)

    off = D['off']
    starts = torch.cat([torch.arange(a, b - crop, device=dev) for a, b in zip(off[:-1], off[1:]) if b - a > crop])
    dstarts = torch.cat([torch.arange(a, b - crop, crop, device=dev) for a, b in zip(Dd['off'][:-1], Dd['off'][1:])])
    SWAP = torch.tensor([1, 0, 2, 3], device=dev)
    VAPN = len(m.VAP_BINS)
    cls = torch.arange(m.VAP_CLASSES, device=dev)
    VAP_PERM = ((cls & ((1 << VAPN) - 1)) << VAPN) | (cls >> VAPN)    # class index after a speaker swap

    def crops(S, idx, aug_rows=None, aug_ch=None):
        span = idx[:, None] + torch.arange(crop, device=dev)
        aspan = (idx[:, None] * FRAME + torch.arange(crop * FRAME, device=dev))
        wav = S['A'][aspan].float()                                    # [B, crop * 1280, 2]
        if aug_rows is not None:
            wa = S['Aa'][aspan].float()
            pick = (aug_rows[:, None] & (torch.arange(2, device=dev)[None] == aug_ch[:, None]))[:, None]
            wav = torch.where(pick, wa, wav)
        tb = {k: v[span] for k, v in S['T'].items()}
        y = {k: v[span] for k, v in S['Y'].items()} if S['Y'] is not None else None
        f = S['F'][span] if S['F'] is not None else None
        return wav, tb, y, f

    def swap(wav, tb, y, f, mask):
        wav = torch.where(mask[:, None, None], wav.flip(-1), wav)
        tb = dict(tb)
        for k in ('act', 'fine', 'yld'):
            if k in tb:
                tb[k] = torch.where(mask.view(-1, *[1] * (tb[k].dim() - 1)), tb[k].flip(2), tb[k])
        for k in ('floor', 'future'):
            tb[k] = torch.where(mask.view(-1, *[1] * (tb[k].dim() - 1)), tb[k][..., SWAP], tb[k])
        tb['vap'] = torch.where(mask[:, None, None], tb['vap'][..., VAP_PERM], tb['vap'])
        if y is not None:
            y = dict(y)
            for k in ('act', 'act_w', 'activity', 'fine', 'fine_w', 'yld', 'yld_w', 'yld_t'):
                y[k] = torch.where(mask.view(-1, *[1] * (y[k].dim() - 1)), y[k].flip(-1), y[k])
            for k in ('floor', 'future'):
                y[k] = torch.where(mask.view(-1, *[1] * (y[k].dim() - 1)), y[k][..., SWAP], y[k])
            v = y['vap']
            y['vap'] = torch.where(mask[:, None], ((v & ((1 << VAPN) - 1)) << VAPN) | (v >> VAPN), v)
        if f is not None:
            f = torch.where(mask[:, None, None, None], f.flip(2), f)
        return wav, tb, y, f

    models, opts, scheds = {}, {}, {}
    for name, cfg in configs.items():
        net = st.build(cfg).to(dev)
        models[name] = net
        opts[name] = torch.optim.AdamW(net.parameters(), lr=cfg.get('lr', lr), weight_decay=cfg.get('wd', 0.05))
        scheds[name] = torch.optim.lr_scheduler.OneCycleLR(opts[name], max_lr=cfg.get('lr', lr), total_steps=steps,
                                                            pct_start=0.05)
        print(name, cfg, f'{st.count_params(net) / 1e6:.2f}M inference params', flush=True)

    stats, stop = [], threading.Event()
    threading.Thread(target=gpu_monitor, args=(stats, stop), daemon=True).start()
    history = {n: [] for n in models}
    best = {n: (float('inf'), None, 0) for n in models}
    tstep = time.time()
    for step in range(1, steps + 1):
        ne = int((torch.rand(batch) < eotb_p).sum()) if De is not None else 0
        idx = starts[torch.randint(len(starts), (batch - ne,), device=dev)]
        rows = torch.rand(batch - ne, device=dev) < bg_p if bg_p > 0 else None
        ch = torch.randint(2, (batch - ne,), device=dev)
        wav, tb, y, f = crops(D, idx, rows, ch)
        if ne:  # append eot-bench rows to the batch
            we, tbe, ye, fe = crops(De, estarts[torch.randint(len(estarts), (ne,), device=dev)])
            wav = torch.cat([wav, we])
            tb = {k: torch.cat([tb[k], tbe[k]]) for k in tb}
            y = {k: torch.cat([y[k], ye[k]]) for k in y}
            f = torch.cat([f, fe])
        wav, tb, y, f = swap(wav, tb, y, f, torch.rand(batch, device=dev) < 0.5)
        wav = wav * 10 ** (torch.empty(batch, 1, 2, device=dev).uniform_(-6, 6) / 20)  # per-channel gain
        for name, net in models.items():
            cfg = configs[name]
            net.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(wav, with_feat=cfg.get('feat', 0) > 0)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = kd_loss(out, tb, cfg.get('kd'))
            if cfg.get('gt', 0) > 0:
                g, gp = m.loss({k: v for k, v in out.items() if k != 'feat'}, y)
                loss = loss + cfg['gt'] * g
                parts.update({f'gt_{k}': v for k, v in gp.items()})
            if cfg.get('feat', 0) > 0:
                fl = ((out['feat'] - (f.float() - f_mu) / f_sd) ** 2).mean()
                loss = loss + cfg['feat'] * fl
                parts['feat'] = fl
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opts[name].step()
            scheds[name].step()
        if step % 100 == 0:
            print(f'step {step} {name}: ' + ' '.join(f'{k} {float(v):.4f}' for k, v in parts.items()), flush=True)
        if step % eval_every == 0 or step == steps:
            for name, net in models.items():
                net.eval()
                tot = {}
                with torch.no_grad():
                    for i in range(0, len(dstarts), 64):
                        wav, tb, y, _ = crops(Dd, dstarts[i:i + 64])
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            out = net(wav)
                        out = {k: v.float() for k, v in out.items()}
                        _, kp = kd_loss(out, tb)
                        _, gp = m.loss(out, y)
                        for k, v in list(kp.items()) + [(f'gt_{k}', v) for k, v in gp.items()]:
                            tot[k] = tot.get(k, 0) + float(v) * len(wav)
                tot = {k: v / len(dstarts) for k, v in tot.items()}
                history[name].append(dict(step=step, **tot))
                key = tot['floor'] + tot['future']
                if key < best[name][0]:
                    best[name] = (key, {k: v.detach().clone() for k, v in net.state_dict().items()}, step)
                print(f'eval {step} {name}: ' + ' '.join(f'{k} {v:.4f}' for k, v in tot.items()), flush=True)
            util = [u for u, _ in stats[-30:]]
            print(f'  {(time.time() - tstep) / step * 1000:.0f} ms/step; GPU util {np.mean(util) if util else -1:.0f}%; '
                  f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
    stop.set()
    os.makedirs(f'{WORK}/runs/{run}', exist_ok=True)
    for name, net in models.items():
        net.load_state_dict(best[name][1])
        torch.save(dict(cfg=configs[name], state=net.state_dict(), teacher=tag, best_step=best[name][2]),
                   f'{WORK}/runs/{run}/{name}.pt')
    json.dump(dict(configs=configs, teacher=tag, train_ids=train_ids, steps=steps, batch=batch, crop=crop,
                   history=history, best_steps={n: b[2] for n, b in best.items()}, wall_s=time.time() - t0),
              open(f'{WORK}/runs/{run}/train.json', 'w'))
    work.commit()
    del D, Dd, De
    torch.cuda.empty_cache()
    if infer:
        _infer_into(run, models, configs)
    return {n: dict(best_step=b[2], best=b[0]) for n, b in best.items()}


def _student_probs(net, cfg, wav, chunk=1000):
    """Exact chunked causal inference: wav [T * 1280, 2] (cuda, float) -> score.py arrays."""
    import numpy as np
    import torch
    import student as st
    ctx = st.context_frames(cfg)
    T = len(wav) // FRAME
    outs = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            lo = max(0, s - ctx)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(wav[lo * FRAME:min(T, s + chunk) * FRAME][None])
            outs.append({k: v[0, s - lo:].float() for k, v in o.items() if k in ('floor', 'future', 'act', 'fine', 'yld')})
    cat = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    res = dict(post=torch.cat([cat['floor'].softmax(-1)[:, None], cat['future'].softmax(-1)], 1),
               silent=cat['act'].softmax(-1)[..., 0], fine=cat['fine'].softmax(-1))
    if 'yld' in cat:
        res['yld'] = cat['yld'].sigmoid()
    return {k: v.cpu().numpy().astype(np.float16) for k, v in res.items()}


def _infer_into(run, models, configs, splits=('oto', 'tbdev')):
    import numpy as np
    import torch
    split = json.load(open(f'{WORK}/split.json'))['splits']
    probs = {}
    for sp in splits:
        ids = split['dev'] if sp == 'oto' else sorted(f[:-4] for f in os.listdir(f'{WORK}/audio16/tbdev')
                                                      if not f.endswith('.tmp.npy'))
        for cid in ids:
            wav = torch.from_numpy(np.load(f'{WORK}/audio16/{sp}/{cid}.npy')).cuda().float()
            for name, net in models.items():
                net.eval()
                for k, v in _student_probs(net, configs[name], wav).items():
                    probs[f'{name}/{sp}/{cid}/{k}'] = v
    np.savez_compressed(f'{WORK}/runs/{run}/probs.npz', **probs)
    work.commit()
    return len(probs)


def load_students(run, names, dev='cuda'):
    import torch
    _paths()
    import student as st
    models, configs = {}, {}
    for n in names:
        ck = torch.load(f'{WORK}/runs/{run}/{n}.pt', map_location=dev, weights_only=False)
        configs[n] = ck['cfg']
        models[n] = st.build(ck['cfg']).to(dev).eval()
        models[n].load_state_dict(ck['state'])
    return models, configs


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=32768, timeout=7200)
def infer(run, names):
    _paths()
    work.reload()
    models, configs = load_students(run, names)
    return _infer_into(run, models, configs)


@app.local_entrypoint()
def train(run: str, configs: str, teacher: str = 'r019s2', steps: int = 6000, batch: int = 48, gpu: str = 'H100',
          lr: float = 1e-3, eval_every: int = 500, no_infer: bool = False, extra: bool = False):
    """--extra adds the otoSpeech conversations outside train and dev (gate and donors) as KD data."""
    cfgs = json.load(open(configs))
    print(train_students.with_options(gpu=gpu).remote(run, cfgs, teacher, steps, batch, 375, eval_every, lr,
                                                      infer=not no_infer, extra_ids=_extra.remote() if extra else ()))


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=600)
def _extra():
    split = json.load(open(f'{WORK}/split.json'))['splits']
    have = sorted(f[:-4] for f in os.listdir(f'{WORK}/labels/oto'))
    return [c for c in have if c not in set(split['train']) | set(split['dev'])]


@app.local_entrypoint()
def reinfer(run: str, names: str):
    print(infer.remote(run, names.split(',')))


# ---- 4. LiveKit eot-bench (same protocol as eotbench/modal_eot.py: user in channel 0, silent agent) ----

@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=3600)
def eot_tracks(run, names, language='en', teacher_tag=''):
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    _paths()
    path = hf_hub_download(EOT_DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=EOT_REVISION)
    table = pq.read_table(path, columns=['id', 'audio'])
    models, configs = load_students(run, names)
    out, t0 = {}, time.time()
    for rid, audio in zip(table.column('id').to_pylist(), table.column('audio').to_pylist()):
        wav, sr = sf.read(io.BytesIO(audio['bytes']), dtype='float32')
        assert sr == 16000, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // FRAME
        x = torch.zeros(T * FRAME, 2, device='cuda')
        x[:, 0] = torch.from_numpy(wav[:T * FRAME]).cuda()
        for name, net in models.items():
            r = _student_probs(net, configs[name], x)
            floor, silent = r['post'][:, 0].astype(np.float32), r['silent'][:, 0].astype(np.float32)
            released = floor[:, 2] + floor[:, 1]
            tracks = dict(eot_q=released * silent)
            if 'yld' in r:
                y = r['yld'][:, 0].astype(np.float32)
                tracks.update(yld_q=y * silent, yld_eq=np.sqrt(y * released) * silent)
            for v, tr_ in tracks.items():
                out[f'{name}/{v}/{rid}'] = np.clip(tr_, 0, 1).astype(np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    print(f'{len(table)} turns in {time.time() - t0:.0f} s')
    return buf.getvalue()


@app.local_entrypoint()
def eot(run: str, names: str, language: str = 'en', out: str = 'tracks.npz'):
    Path(out).write_bytes(eot_tracks.remote(run, names.split(','), language))
    print('wrote', out)
