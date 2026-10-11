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
  audio16/pool/{sid}.npy           float16 [T * 1280] mono, the unlabeled pool (pool.py)
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

def _np(x):
    import numpy as np
    return np.ascontiguousarray(x)


def _load(split, cids, tag, device, labels=True, aug=False, feats=False, vad=False, mono=False):
    """One data source, frames concatenated over conversations (crops never cross a boundary):
      A  [N * 1280, C] float16 audio on `device` (C = 1 with mono, the pool), Aa its background copy
      T  teacher logits, Y gold labels (+ pause targets) on `device`
      F  the teacher's FastConformer frames, lazily: one memory map per conversation
      V / Va  Silero VAD per frame, uint8 [N, C, 2] ({WORK}/vadf/{split}[_aug]/{cid}.npy)
    `off` holds the frame offsets."""
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
        if feats:
            T = min(T, np.load(f'{WORK}/feats_asr/{split}/{cid}.npy', mmap_mode='r').shape[0])
        return T

    def read(cid, T):
        a = np.load(f'{WORK}/audio16/{split}/{cid}.npy', mmap_mode='r')[:T * FRAME]
        r = dict(a=a[:, None] if a.ndim == 1 else a)
        if aug:
            p = f'{WORK}/audio16_aug/{split}/{cid}.npy'
            r['aug'] = np.load(p, mmap_mode='r')[:T * FRAME] if os.path.exists(p) else r['a']
        with np.load(f'{WORK}/teacher/{tag}/{split}/{cid}.npz') as z:
            r['t'] = {k: z[k][:T] for k in z.files}
        if vad:
            v = np.load(f'{WORK}/vadf/{split}/{cid}.npy')[:T]
            r['v'] = v[:, None] if v.ndim == 2 else v
            if aug:
                p = f'{WORK}/vadf/{split}_aug/{cid}.npy'
                r['va'] = np.load(p)[:T] if os.path.exists(p) else r['v']
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
        C = 1 if mono else 2
        A = torch.empty((N * FRAME, C), dtype=torch.float16, device=device)
        Aa = torch.empty((N * FRAME, C), dtype=torch.float16, device=device) if aug else None
        V = torch.empty((N, C, 2), dtype=torch.uint8, device=device) if vad else None
        Va = torch.empty((N, C, 2), dtype=torch.uint8, device=device) if vad and aug else None
        Tch, Y = {}, []
        for a, r in zip(off[:-1], pool.map(read, cids, lengths)):
            T = len(r['a']) // FRAME
            A[a * FRAME:(a + T) * FRAME] = torch.from_numpy(_np(r['a'])).to(device)
            if aug:
                Aa[a * FRAME:(a + T) * FRAME] = torch.from_numpy(_np(r['aug'])).to(device)
            if vad:
                V[a:a + T] = torch.from_numpy(_np(r['v'])).to(device)
                if aug:
                    Va[a:a + T] = torch.from_numpy(_np(r['va'])).to(device)
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
    Fz = [np.load(f'{WORK}/feats_asr/{split}/{c}.npy', mmap_mode='r') for c in cids] if feats else None
    return dict(A=A, Aa=Aa, F=Fz, T=Tch, Y=Yt, V=V, Va=Va, off=off, n=len(cids))


def _load_pool(cids, tag, device, feats=False, vad=False):
    """Unlabeled pool (pool.py): mono user audio, the teacher's logits (agent channel: digital silence)."""
    return _load('pool', cids, tag, device, labels=False, feats=feats, vad=vad, mono=True)


def _starts(off, crop, step=1):
    import torch
    return torch.cat([torch.arange(a, b - crop, step) for a, b in zip(off[:-1], off[1:]) if b - a > crop])


def _gather(x, starts, crop, scale=1):
    """Rows x[s * scale:(s + crop) * scale] for each start (plain slices: fast on CPU and GPU)."""
    import torch
    return torch.stack([x[s * scale:(s + crop) * scale] for s in starts])


def _gather_feats(S, starts, crop):
    """The lazily mapped teacher frames of each crop: [B, crop, 2, 1024] float16 (CPU)."""
    import bisect
    import numpy as np
    import torch
    out = np.empty((len(starts), crop, 2, 1024), np.float16)
    for j, s in enumerate(starts):
        i = bisect.bisect_right(S['off'], s) - 1
        lo = s - S['off'][i]
        out[j] = S['F'][i][lo:lo + crop]
    return torch.from_numpy(out)


def _crops(S, starts, crop, dev, aug_rows=None, aug_ch=None, feats=False):
    """Crops of one source moved to `dev`: wav [B, crop * 1280, 2] float, teacher dict, labels dict,
    teacher frames, VAD frames [B, crop, 2, 2]. Mono sources get a silent second channel."""
    import torch
    starts = [int(s) for s in starts]
    wav = _gather(S['A'], starts, crop, FRAME).to(dev, non_blocking=True).float()
    v = _gather(S['V'], starts, crop).to(dev) if S['V'] is not None else None
    if aug_rows is not None:
        pick = (aug_rows[:, None] & (torch.arange(wav.shape[-1], device=dev)[None] == aug_ch[:, None]))
        wa = _gather(S['Aa'], starts, crop, FRAME).to(dev, non_blocking=True).float()
        wav = torch.where(pick[:, None], wa, wav)
        if v is not None:
            v = torch.where(pick[:, None, :, None], _gather(S['Va'], starts, crop).to(dev), v)
    if wav.shape[-1] == 1:
        wav = torch.cat([wav, torch.zeros_like(wav)], -1)
        if v is not None:
            v = torch.cat([v, torch.zeros_like(v)], 2)
    tb = {k: _gather(t, starts, crop).to(dev, non_blocking=True) for k, t in S['T'].items()}
    y = {k: _gather(t, starts, crop).to(dev, non_blocking=True) for k, t in S['Y'].items()} if S['Y'] is not None else None
    f = _gather_feats(S, starts, crop).to(dev, non_blocking=True) if feats else None
    return wav, tb, y, f, v


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


def _cat(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, dict):
        return {k: _cat(a[k], b[k]) for k in a}
    import torch
    return torch.cat([a, b])


@app.function(image=image, volumes=VOLUMES, gpu='H100', cpu=8, memory=65536, timeout=4 * 3600)
def train_students(run, configs, tag, steps=6000, batch=48, crop=375, eval_every=500, lr=1e-3, seed=0,
                   infer=True, extra_ids=(), data_device='cuda', select='oto', keep_all=False):
    """Train every student in `configs` on shared batches.

    Batch rows: otoSpeech crops (one channel from the background copy with probability bg_aug),
    eot-bench voice-agent rows (eotb_aug), unlabeled pool rows (pool_aug). Per-config options:
    gt / feat loss weights, acoustic (augment.acoustic probability, student input only), vad_in
    (Silero VAD per frame as input; needs {WORK}/vadf, see colab.py vad), arch 'fc' (student_fc.py).
    data_device 'cpu' keeps the data in host RAM (a 40 GB GPU); select 'ood' holds out 10% of the
    eot-bench sessions and 2.5% of the pool and picks each student's checkpoint by its summed KD loss
    on otoSpeech dev and those held-out sets (default: otoSpeech dev only). keep_all saves every
    evaluated checkpoint to runs/{run}/ckpt/.
    """
    import queue
    import threading
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import numpy as np
    import torch
    _paths()
    import augment
    import model as m
    import student as st
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    work.reload()
    split = json.load(open(f'{WORK}/split.json'))['splits']
    have = {f[:-4] for f in os.listdir(f'{WORK}/teacher/{tag}/oto')}
    train_ids = [c for c in split['train'] if c in have] + [c for c in extra_ids if c in have]
    dev_ids = [c for c in split['dev'] if c in have]

    def shared(key, default=0.0):
        vals = {json.dumps(c.get(key, default)) for c in configs.values()}
        assert len(vals) == 1, f'students in one run share batches: {key} differs'
        return configs[next(iter(configs))].get(key, default)

    bg_p = max(c.get('bg_aug', 0.0) for c in configs.values())
    eotb_p, pool_p = shared('eotb_aug'), shared('pool_aug')
    use_feat = any(c.get('feat', 0) > 0 for c in configs.values())
    use_vad = any(c.get('vad_in') for c in configs.values())
    t0 = time.time()
    D = _load('oto', train_ids, tag, data_device, aug=bg_p > 0, feats=use_feat, vad=use_vad)
    Dd = _load('oto', dev_ids, tag, data_device, vad=use_vad)
    evals = {'oto': (Dd, _starts(Dd['off'], crop, crop))}
    De = Dp = None
    if eotb_p > 0:  # voice-agent user turns (latency/eotb_data.py sessions), silent agent channel
        eids = sorted(f[:-4] for f in os.listdir(f'{WORK}/teacher/{tag}/eotb'))
        held = eids[::10] if select == 'ood' else []
        De = _load('eotb', [e for e in eids if e not in held], tag, data_device, feats=use_feat, vad=use_vad)
        estarts = _starts(De['off'], crop)
        if held:
            Dh = _load('eotb', held, tag, data_device, vad=use_vad)
            evals['eotb_ho'] = (Dh, _starts(Dh['off'], crop, crop))
        print(f'eot-bench sessions: {len(eids) - len(held)} train, {len(held)} held out', flush=True)
    if pool_p > 0:  # unlabeled pool (pool.py), silent agent channel, teacher logits only
        pids = sorted(f[:-4] for f in os.listdir(f'{WORK}/teacher/{tag}/pool'))
        held = pids[::40] if select == 'ood' else []
        Dp = _load_pool([p for p in pids if p not in held], tag, data_device, feats=use_feat, vad=use_vad)
        pstarts = _starts(Dp['off'], crop)
        if held:
            Dh = _load_pool(held, tag, data_device, vad=use_vad)
            evals['pool_ho'] = (Dh, _starts(Dh['off'], crop, crop)[::4])
        print(f'pool sessions: {Dp["n"]} train ({Dp["off"][-1] * 0.08 / 3600:.1f} h), {len(held)} held out', flush=True)
    pool_bg = bg_p > 0 and not use_vad  # VAD inputs are precomputed on clean pool audio
    print(f'loaded {len(train_ids)} train ({D["off"][-1]} frames) / {len(dev_ids)} dev in {time.time() - t0:.0f}s '
          f'on {data_device}; VRAM {torch.cuda.memory_allocated() / 2**30 if dev == "cuda" else 0:.1f} GiB', flush=True)
    if use_feat:  # standardization of the teacher's encoder frames (feature distillation target)
        s_all = _starts(D['off'], crop)
        sample = _gather_feats(D, [int(s) for s in s_all[torch.randint(len(s_all), (64,))]], crop)
        sample = sample.float().flatten(0, 2).to(dev)
        f_mu, f_sd = sample.mean(0), sample.std(0).clamp_min(1e-3)

    starts = _starts(D['off'], crop)
    SWAP = torch.tensor([1, 0, 2, 3], device=dev)
    VAPN = len(m.VAP_BINS)
    cls = torch.arange(m.VAP_CLASSES, device=dev)
    VAP_PERM = ((cls & ((1 << VAPN) - 1)) << VAPN) | (cls >> VAPN)    # class index after a speaker swap

    def swap(wav, tb, y, f, v, mask):
        def w(x):
            return mask.view(-1, *[1] * (x.dim() - 1))
        wav = torch.where(mask[:, None, None], wav.flip(-1), wav)
        tb = dict(tb)
        for k in ('act', 'fine', 'yld'):
            if k in tb:
                tb[k] = torch.where(w(tb[k]), tb[k].flip(2), tb[k])
        for k in ('floor', 'future'):
            tb[k] = torch.where(w(tb[k]), tb[k][..., SWAP], tb[k])
        tb['vap'] = torch.where(mask[:, None, None], tb['vap'][..., VAP_PERM], tb['vap'])
        if y is not None:
            y = dict(y)
            for k in ('act', 'act_w', 'activity', 'fine', 'fine_w', 'yld', 'yld_w', 'yld_t'):
                y[k] = torch.where(w(y[k]), y[k].flip(-1), y[k])
            for k in ('floor', 'future'):
                y[k] = torch.where(w(y[k]), y[k][..., SWAP], y[k])
            u = y['vap']
            y['vap'] = torch.where(mask[:, None], ((u & ((1 << VAPN) - 1)) << VAPN) | (u >> VAPN), u)
        if f is not None:
            f = torch.where(mask[:, None, None, None], f.flip(2), f)
        if v is not None:
            v = torch.where(mask[:, None, None, None], v.flip(2), v)
        return wav, tb, y, f, v

    def make_batch():
        u = torch.rand(batch)
        ne = int((u < eotb_p).sum()) if De is not None else 0
        npl = int(((u >= eotb_p) & (u < eotb_p + pool_p)).sum()) if Dp is not None else 0
        nl = batch - npl                                               # rows with gold labels
        idx = starts[torch.randint(len(starts), (nl - ne,))]
        rows = torch.rand(nl - ne, device=dev) < bg_p if bg_p > 0 else None
        ch = torch.randint(2, (nl - ne,), device=dev)
        wav, tb, y, f, v = _crops(D, idx, crop, dev, rows, ch, feats=use_feat)
        if ne:
            we, tbe, ye, fe, ve = _crops(De, estarts[torch.randint(len(estarts), (ne,))], crop, dev, feats=use_feat)
            wav, tb, y, f, v = torch.cat([wav, we]), _cat(tb, tbe), _cat(y, ye), _cat(f, fe), _cat(v, ve)
        wav, tb, y, f, v = swap(wav, tb, y, f, v, torch.rand(nl, device=dev) < 0.5)
        if npl:  # unlabeled pool rows last: user in channel 0, digital silence in channel 1
            wp, tbp, _, fp, vp = _crops(Dp, pstarts[torch.randint(len(pstarts), (npl,))], crop, dev, feats=use_feat)
            if pool_bg:  # background: the residual (augmented - clean) of a random otoSpeech crop
                bidx = [int(s) for s in starts[torch.randint(len(starts), (npl,))]]
                res = (_gather(D['Aa'], bidx, crop, FRAME).to(dev).float() - _gather(D['A'], bidx, crop, FRAME).to(dev).float())
                bch = torch.randint(2, (npl,), device=dev)
                res = torch.where(bch[:, None] == 0, res[..., 0], res[..., 1])
                wp[..., 0] += res * (torch.rand(npl, 1, device=dev) < bg_p)
            wp, tbp, _, fp, vp = swap(wp, tbp, None, fp, vp, torch.rand(npl, device=dev) < 0.5)
            wav, tb, f, v = torch.cat([wav, wp]), _cat(tb, tbp), _cat(f, fp), _cat(v, vp)
        wav = wav * 10 ** (torch.empty(batch, 1, 2, device=dev).uniform_(-6, 6) / 20)  # per-channel gain
        return wav, tb, y, f, v, nl

    q = queue.Queue(maxsize=3)
    stop = threading.Event()

    def producer():
        while not stop.is_set():
            b = make_batch()
            while not stop.is_set():
                try:
                    q.put(b, timeout=1)
                    break
                except queue.Full:
                    pass

    workers = [threading.Thread(target=producer, daemon=True) for _ in range(2 if data_device == 'cpu' else 1)]
    for t in workers:
        t.start()

    models, opts, scheds = {}, {}, {}
    for name, cfg in configs.items():
        net = st.build(cfg).to(dev)
        models[name] = net
        enc = set(map(id, net.encoder_parameters())) if hasattr(net, 'encoder_parameters') else set()
        groups = [dict(params=[p for p in net.parameters() if id(p) not in enc], lr=cfg.get('lr', lr))]
        if enc:
            groups.append(dict(params=net.encoder_parameters(), lr=cfg.get('enc_lr', 1e-4)))
        opts[name] = torch.optim.AdamW(groups, weight_decay=cfg.get('wd', 0.05))
        scheds[name] = torch.optim.lr_scheduler.OneCycleLR(opts[name], max_lr=[g['lr'] for g in groups],
                                                            total_steps=steps, pct_start=0.05)
        print(name, cfg, f'{st.count_params(net) / 1e6:.2f}M inference params', flush=True)

    stats, mon_stop = [], threading.Event()
    if dev == 'cuda':
        threading.Thread(target=gpu_monitor, args=(stats, mon_stop), daemon=True).start()
    history = {n: [] for n in models}
    best = {n: (float('inf'), None, 0) for n in models}
    os.makedirs(f'{WORK}/runs/{run}' + ('/ckpt' if keep_all else ''), exist_ok=True)
    tstep = time.time()
    for step in range(1, steps + 1):
        wav, tb, y, f, v, nl = q.get()
        for name, net in models.items():
            cfg = configs[name]
            net.train()
            x = augment.acoustic(wav, cfg['acoustic']) if cfg.get('acoustic', 0) > 0 else wav
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                out = net(x, with_feat=cfg.get('feat', 0) > 0, vad=v)
            out = {k: o.float() for k, o in out.items()}
            loss, parts = kd_loss(out, tb, cfg.get('kd'))
            if cfg.get('gt', 0) > 0:
                g, gp = m.loss({k: o[:nl] for k, o in out.items() if k != 'feat'}, y)
                loss = loss + cfg['gt'] * g
                parts.update({f'gt_{k}': o for k, o in gp.items()})
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
            print(f'step {step} {name}: ' + ' '.join(f'{k} {float(o):.4f}' for k, o in parts.items()), flush=True)
        if step % eval_every == 0 or step == steps:
            for name, net in models.items():
                net.eval()
                rec = dict(step=step)
                for ev, (S, ds) in evals.items():
                    tot, n = {}, 0
                    with torch.no_grad():
                        for i in range(0, len(ds), 64):
                            wv, tbv, yv, _, vv = _crops(S, ds[i:i + 64], crop, dev)
                            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                                out = net(wv, vad=vv)
                            out = {k: o.float() for k, o in out.items()}
                            _, kp = kd_loss(out, tbv)
                            items = list(kp.items())
                            if yv is not None:
                                items += [(f'gt_{k}', o) for k, o in m.loss(out, yv)[1].items()]
                            for k, o in items:
                                tot[k] = tot.get(k, 0) + float(o) * len(wv)
                            n += len(wv)
                    rec.update({(k if ev == 'oto' else f'{ev}_{k}'): o / n for k, o in tot.items()})
                history[name].append(rec)
                key = sum(rec[p + 'floor'] + rec[p + 'future'] for p in
                          [''] + [f'{e}_' for e in evals if e != 'oto'])
                if key < best[name][0]:
                    best[name] = (key, {k: o.detach().clone() for k, o in net.state_dict().items()}, step)
                if keep_all:
                    torch.save(dict(cfg=configs[name], state=net.state_dict(), teacher=tag, step=step),
                               f'{WORK}/runs/{run}/ckpt/{name}_{step:05d}.pt')
                print(f'eval {step} {name}: select {key:.4f} ' +
                      ' '.join(f'{k} {o:.4f}' for k, o in rec.items() if 'floor' in k or 'future' in k), flush=True)
            util = [u for u, _ in stats[-30:]]
            print(f'  {(time.time() - tstep) / step * 1000:.0f} ms/step; GPU util {np.mean(util) if util else -1:.0f}%; '
                  f'peak {torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0:.1f} GiB', flush=True)
    stop.set()
    mon_stop.set()
    for name, net in models.items():
        net.load_state_dict(best[name][1])
        torch.save(dict(cfg=configs[name], state=net.state_dict(), teacher=tag, best_step=best[name][2]),
                   f'{WORK}/runs/{run}/{name}.pt')
    json.dump(dict(configs=configs, teacher=tag, train_ids=train_ids, steps=steps, batch=batch, crop=crop,
                   select=select, history=history, best_steps={n: b[2] for n, b in best.items()},
                   wall_s=time.time() - t0),
              open(f'{WORK}/runs/{run}/train.json', 'w'))
    work.commit()
    del D, Dd, De, Dp, evals
    if dev == 'cuda':
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
    vad = None
    if cfg.get('vad_in'):  # Silero on the same audio, per 80 ms frame (CPU)
        import augment
        vad = torch.from_numpy(augment.vad_frames(wav[:T * FRAME].float().cpu().numpy())).to(wav.device)
    outs = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            lo = max(0, s - ctx)
            with torch.autocast(wav.device.type, dtype=torch.bfloat16, enabled=wav.device.type == 'cuda'):
                o = net(wav[lo * FRAME:min(T, s + chunk) * FRAME][None],
                        vad=vad[lo:min(T, s + chunk)][None] if vad is not None else None)
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
            wav = torch.from_numpy(np.load(f'{WORK}/audio16/{sp}/{cid}.npy')).to(_device()).float()
            for name, net in models.items():
                net.eval()
                for k, v in _student_probs(net, configs[name], wav).items():
                    probs[f'{name}/{sp}/{cid}/{k}'] = v
    np.savez_compressed(f'{WORK}/runs/{run}/probs.npz', **probs)
    work.commit()
    return len(probs)


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_students(run, names, dev=None):
    import torch
    dev = dev or _device()
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
        x = torch.zeros(T * FRAME, 2, device=_device())
        x[:, 0] = torch.from_numpy(wav[:T * FRAME]).to(x.device)
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
