"""Train several floor models in lockstep on GPU-resident frozen features, then infer.

All train features live in VRAM as one fp16 tensor; batches are random 30 s crops
gathered by index on the GPU, so there is no data loader. Every config sees the same
batches, which makes a grid one data load and keeps the GPU busy. After training,
each model runs exact chunked causal inference on oto dev and TurnBench dev and writes
per-frame EOT/INT probabilities (12.5 Hz grid) for score.py.

    modal run train.py::main --run r001 --configs configs/r001.json
"""
import json
import time

import modal

from common import VOLUMES, gpu_image, gpu_monitor, setup_path, work

app = modal.App('ssl-turn-train')
TAPS = (7, 15, 23, 31)  # layout of encode.py features: 4 x 1280 taps, then 768 final
TAP_DIM, FINAL_DIM = 1280, 768
LOADED = list(TAPS)  # taps actually kept in VRAM for this run (set by train/infer)
FEAT_DIR = 'feats'   # 'feats' (Cat) or 'feats_mtd' (MOSS-Transcribe-Diarize, 4096-d, no taps)
FUSE_MTD = False     # Cat columns + MTD 4096-d appended (config feats='cat+mtd')
WARM = 0             # context-only frames before each training crop (encoder-tuning runs: CatTop window)
CAT_DIR = '/work/models/cat'
_CAT = None


def columns(taps):
    """Feature columns for the given taps (in TAPS order) followed by the final output."""
    import numpy as np
    cols = [np.arange(TAPS.index(t) * TAP_DIM, (TAPS.index(t) + 1) * TAP_DIM) for t in taps]
    cols.append(np.arange(len(TAPS) * TAP_DIM, len(TAPS) * TAP_DIM + FINAL_DIM))
    return np.concatenate(cols)


def load_split(split, cids, device, cols=None, workers=16):
    """Read per-conversation features with parallel threads (volume reads are I/O bound)
    straight into one preallocated GPU tensor; peak VRAM is the final size, not 2x."""
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor

    def length(cid):
        T = np.load(f'/work/{FEAT_DIR}/{split}/{cid}.npy', mmap_mode='r').shape[0]
        if FUSE_MTD:
            T = min(T, np.load(f'/work/feats_mtd/{split}/{cid}.npy', mmap_mode='r').shape[0])
        if split == 'oto':
            with np.load(f'/work/labels/oto/{cid}.npz') as z:
                T = min(T, len(z['floor']))
        return T

    def read(cid, T):
        f = np.load(f'/work/{FEAT_DIR}/{split}/{cid}.npy', mmap_mode='r')
        lab = None
        if split == 'oto':
            z = np.load(f'/work/labels/oto/{cid}.npz')
            lab = {k: z[k][:T] for k in ('floor', 'floor_w', 'act', 'act_w', 'future', 'future_w', 'activity')}
            if 'fine' in z.files:
                lab['fine'], lab['fine_w'] = z['fine'][:T], z['act_w'][:T]
        x = np.ascontiguousarray(f[:T]) if cols is None else np.ascontiguousarray(f[:T][..., cols])
        if FUSE_MTD:  # append the 4096-d MOSS-Transcribe-Diarize features after the Cat columns
            g = np.load(f'/work/feats_mtd/{split}/{cid}.npy', mmap_mode='r')
            x = np.concatenate([x, np.ascontiguousarray(g[:T])], -1)
        return x, lab

    with ThreadPoolExecutor(workers) as pool:
        lengths = list(pool.map(length, cids))
        offsets = np.concatenate([[0], np.cumsum(lengths)]).tolist()
        dim = len(cols) if cols is not None else np.load(f'/work/{FEAT_DIR}/{split}/{cids[0]}.npy', mmap_mode='r').shape[-1]
        dim += 4096 if FUSE_MTD else 0
        X = torch.empty((offsets[-1], 2, dim), dtype=torch.float16, device=device)
        labs = []
        for a, (x, lab) in zip(offsets[:-1], pool.map(read, cids, lengths)):
            X[a:a + len(x)] = torch.from_numpy(x).to(device)
            if lab is not None:
                labs.append(lab)
    return X, offsets, labs


def stack_labels(lab, device):
    import numpy as np
    import torch
    setup_path()
    import model as m
    out = {k: torch.from_numpy(np.concatenate([l[k] for l in lab])).to(device) for k in lab[0]}
    vap, valid = zip(*[m.vap_labels(l['activity']) for l in lab])
    out['vap'] = torch.from_numpy(np.concatenate(vap)).to(device)
    out['vap_valid'] = torch.from_numpy(np.concatenate(valid)).to(device)
    return out


def select_inputs(x, cfg):
    """x [..., 2, 5888] fp16 -> (taps [..., 2, L, 1280] or None, final [..., 2, 768] or None)."""
    taps = None
    if cfg.get('taps'):
        idx = [LOADED.index(t) for t in cfg['taps']]
        all_taps = x[..., :len(LOADED) * TAP_DIM].unflatten(-1, (len(LOADED), TAP_DIM))
        taps = all_taps[..., idx, :]
    final = x[..., len(LOADED) * TAP_DIM:] if cfg.get('final', True) else None
    return taps, final


def run_net(net, cfg, x, warm=0):
    """Forward one model on loaded features x [B, T, 2, D]; outputs cover frames [warm, T)."""
    tune = cfg.get('tune')
    if tune:
        first = tune.get('first', 16)
        i = LOADED.index(first - 1)
        return net(x[..., i * TAP_DIM:(i + 1) * TAP_DIM], warm)
    taps, final = select_inputs(x[:, warm:], cfg)
    return net(taps, final)


def model_context(cfg):
    """Left context (frames) for exact chunked inference: head windows (+ CatTop windows)."""
    ctx = cfg.get('layers', 4) * int(round(cfg.get('window_s', 20.0) / 0.08))
    if cfg.get('tune'):
        setup_path()
        import cat_top
        ctx += (32 - cfg['tune'].get('first', 16)) * (cat_top.WINDOW - 1)
    return ctx


def infer_all(models, configs, splits, dev):
    """Exact chunked causal inference: each 1000-frame chunk carries enough left context to
    cover the stacked attention windows (layers x window frames). Exports floor posteriors
    (now + projections) and per-speaker p(SILENT) from the act head."""
    import numpy as np
    import torch
    out = {}
    for name, net in models.items():
        net.eval()
        cfg = configs[name]
        ctx = model_context(cfg)
        for split, Xs, offs, ids in splits:
            for cid, a, b in zip(ids, offs[:-1], offs[1:]):
                outs = []
                with torch.no_grad():
                    for s in range(a, b, 1000):
                        lo = max(a, s - ctx)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            o = run_net(net, cfg, Xs[lo:min(b, s + 1000)][None])
                        outs.append({k: v[0, s - lo:].float() for k, v in o.items()
                                     if k in ('floor', 'future', 'act', 'fine')})
                floor = torch.cat([o['floor'] for o in outs]).softmax(-1)
                future = torch.cat([o['future'] for o in outs]).softmax(-1)
                silent = torch.cat([o['act'] for o in outs]).softmax(-1)[..., 0]  # [T, 2] p(SILENT)
                out[f'{name}/{split}/{cid}/post'] = torch.cat([floor[:, None], future], 1).cpu().numpy().astype(np.float16)
                out[f'{name}/{split}/{cid}/silent'] = silent.cpu().numpy().astype(np.float16)
                if 'fine' in outs[0]:  # [T, 2, len(FINE)] fine-label posteriors
                    out[f'{name}/{split}/{cid}/fine'] = torch.cat([o['fine'] for o in outs]).softmax(-1) \
                        .cpu().numpy().astype(np.float16)
    return out


@app.function(image=gpu_image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=3600)
def infer(run, names, out_run=None):
    """Re-run inference from saved checkpoints (no training) and write probs.npz."""
    import os
    import numpy as np
    import torch
    dev = 'cuda'
    split = json.load(open('/work/split.json'))['splits']
    have = {f[:-4] for f in os.listdir('/work/feats/oto')}
    dev_ids = [c for c in split['dev'] if c in have] if use_dev else []
    tb_ids = sorted(f[:-4] for f in os.listdir('/work/feats/tbdev'))
    global LOADED, FEAT_DIR
    models, configs = {}, {}
    for n in names:
        ck = torch.load(f'/work/runs/{run}/{n}.pt', map_location=dev)
        configs[n] = ck['cfg']
        models[n] = build_model(ck['cfg']).to(dev)
        models[n].load_state_dict(ck['state'], strict=not ck['cfg'].get('tune'))  # tuned: trainable params only
    mtd = any(c.get('feats') == 'mtd' for c in configs.values())
    FEAT_DIR = 'feats_mtd' if mtd else 'feats'
    LOADED = [t for t in TAPS if any(t in (c.get('taps') or []) for c in configs.values())]
    cols = None if mtd else columns(LOADED)
    Xd, offd, _ = load_split('oto', dev_ids, dev, cols)
    probs = infer_all(models, configs, (('oto', Xd, offd, dev_ids),), dev)
    del Xd
    Xt, offt, _ = load_split('tbdev', tb_ids, dev, cols)
    probs.update(infer_all(models, configs, (('tbdev', Xt, offt, tb_ids),), dev))
    out_run = out_run or run
    os.makedirs(f'/work/runs/{out_run}', exist_ok=True)
    np.savez_compressed(f'/work/runs/{out_run}/probs.npz', **probs)
    work.commit()
    return len(probs)


def load_cat():
    """The Cat encoder (CPU, fp32) whose top stage CatTop copies; built once per container."""
    global _CAT
    if _CAT is None:
        setup_path()
        import cat_encoder as ce
        _CAT = ce.build(CAT_DIR, f'{CAT_DIR}/cat_encoder.safetensors', taps=())
    return _CAT


def build_model(cfg):
    import torch
    head = build_head(cfg)
    tune = cfg.get('tune')
    if not tune:
        return head
    import cat_top
    first = tune.get('first', 16)
    top = cat_top.CatTop(load_cat(), first=first, taps=[t for t in cfg['taps'] if t >= first],
                         rank=tune.get('rank', 16), alpha=tune.get('alpha', 32.0), dropout=tune.get('dropout', 0.0),
                         train_norms=tune.get('norms', False), checkpoint=tune.get('checkpoint', False),
                         base_dtype=torch.bfloat16)
    return cat_top.Tuned(top, head, cfg['taps'])


def build_head(cfg):
    setup_path()
    import model as m
    final_dim = {'mtd': 4096, 'cat+mtd': FINAL_DIM + 4096}.get(cfg.get('feats'), FINAL_DIM)
    return m.TurnModel(tap_layers=len(cfg.get('taps') or []), tap_dim=TAP_DIM,
                       final_dim=final_dim if cfg.get('final', True) else 0,
                       dim=cfg.get('dim', 256), heads=cfg.get('heads', 4), layers=cfg.get('layers', 4),
                       window_s=cfg.get('window_s', 20.0), dropout=cfg.get('dropout', 0.1))


@app.function(image=gpu_image, volumes=VOLUMES, gpu='A100', cpu=4, memory=16384, timeout=5400)
def train(run, configs, n_train=32, steps=1500, batch=64, crop=375, eval_every=250, seed=0, extra=False,
          use_dev=True, train_from=None, infer=True):
    import os, threading
    import numpy as np
    import torch
    setup_path()
    import labels as lb
    import model as m
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    split = json.load(open('/work/split.json'))['splits']
    have = {f[:-4] for f in os.listdir('/work/feats/oto')}
    train_ids = [c for c in split['train'] if c in have][:n_train]
    if extra:  # labeled data-scaling ablation: gate-free cross-partition conversations
        train_ids += [c for c in json.load(open('/work/extra_no_gate.json')) if c in have]
    global LOADED, FEAT_DIR, FUSE_MTD, WARM
    setup_path()
    import cat_top
    WARM = cat_top.WINDOW if any(c.get('tune') for c in configs.values()) else 0
    mtd = any(c.get('feats') == 'mtd' for c in configs.values())
    FUSE_MTD = any(c.get('feats') == 'cat+mtd' for c in configs.values())
    FEAT_DIR = 'feats_mtd' if mtd else 'feats'
    have = {f[:-4] for f in os.listdir(f'/work/{FEAT_DIR}/oto') if not f.endswith('.tmp.npy')}
    if train_from:  # pin the exact conversation list (e.g. to match another backbone's subset)
        train_ids = [c for c in json.load(open(train_from)) if c in have]
    train_ids = [c for c in train_ids if c in have]
    need = {t for c in configs.values() for t in (c.get('taps') or [])}
    need |= {c['tune'].get('first', 16) - 1 for c in configs.values() if c.get('tune')}
    LOADED = [t for t in TAPS if t in need]
    cols = None if mtd else columns(LOADED)
    dev_ids = [c for c in split['dev'] if c in have] if use_dev else []
    tb_ids = sorted(f[:-4] for f in os.listdir('/work/feats/tbdev'))
    t0 = time.time()
    X, off, lab = load_split('oto', train_ids, dev, cols)
    Y = stack_labels(lab, dev)
    if dev_ids:
        Xd, offd, labd = load_split('oto', dev_ids, dev, cols)
        Yd = stack_labels(labd, dev)
    print(f'loaded {len(train_ids)} train ({len(X)} frames) / {len(dev_ids)} dev in {time.time() - t0:.0f}s; '
          f'VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)

    # Inverse-frequency class weights for the fine head (per config: weight = freq^-alpha,
    # normalized to mean 1 over training frames, clipped at 20x).
    fine_cw = None
    if 'fine' in Y:
        counts = torch.bincount(Y['fine'].flatten(), minlength=len(lb.FINE)).float().clamp_min(1)
        freq = counts / counts.sum()
        fine_cw = {}
        for name, cfg in configs.items():
            a = cfg.get('fine_balance', 0.0)
            w = freq.pow(-a)
            w = w / (w * freq).sum()
            fine_cw[name] = w.clamp(max=20.0)
        print('fine class frames', counts.long().tolist(), flush=True)

    # Valid crop starts: within one conversation.
    # Valid crop starts (first labelled frame): WARM context frames before it, all in one conversation.
    starts = torch.cat([torch.arange(a + WARM, b - crop, device=dev) for a, b in zip(off[:-1], off[1:])
                        if b - a > crop + WARM])
    models, opts, scheds = {}, {}, {}
    for name, cfg in configs.items():
        net = build_model(cfg).to(dev)
        models[name] = net
        groups = [dict(params=[p for n, p in net.named_parameters() if p.requires_grad and not n.startswith('top.')],
                       lr=cfg.get('lr', 3e-4), weight_decay=cfg.get('wd', 0.05))]
        if cfg.get('tune'):  # LoRA (+ norms) of the encoder top: own learning rate, no weight decay
            groups.append(dict(params=[p for n, p in net.named_parameters() if p.requires_grad and n.startswith('top.')],
                               lr=cfg['tune'].get('lr', 1e-4), weight_decay=0.0))
        opts[name] = torch.optim.AdamW(groups)
        scheds[name] = torch.optim.lr_scheduler.OneCycleLR(opts[name], max_lr=[g['lr'] for g in groups],
                                                            total_steps=steps, pct_start=0.1)
        n_train_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(name, cfg, f'{n_train_p / 1e6:.2f}M trainable / {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params',
              flush=True)

    def crops(Xs, Ys, idx):
        """Features for [idx - WARM, idx + crop), labels for [idx, idx + crop)."""
        span = idx[:, None] + torch.arange(-WARM, crop, device=dev)
        b = {k: v[span[:, WARM:]] for k, v in Ys.items()}
        return Xs[span], b

    SWAP = torch.tensor([1, 0, 2, 3], device=dev)  # HELD_0 <-> HELD_1
    VAPN = len(m.VAP_BINS)

    def swap_speakers(xb, yb, mask):
        """Channel-swap augmentation on rows in `mask`: features, acts, floor holders, VAP bits."""
        xb = torch.where(mask[:, None, None, None], xb.flip(2), xb)
        out = dict(yb)
        for k in [k for k in ('act', 'act_w', 'activity', 'fine', 'fine_w') if k in yb]:
            out[k] = torch.where(mask.view(-1, *[1] * (yb[k].dim() - 1)), yb[k].flip(-1), yb[k])
        for k in ('floor', 'future'):
            out[k] = torch.where(mask.view(-1, *[1] * (yb[k].dim() - 1)), yb[k][..., SWAP], yb[k])
        v = yb['vap']
        lo, hi = v & ((1 << VAPN) - 1), v >> VAPN
        out['vap'] = torch.where(mask[:, None], (lo << VAPN) | hi, v)
        return xb, out

    def regularize(xb, cfg):
        if cfg.get('feat_dropout', 0) > 0:
            keep = torch.rand(xb.shape[:-1] + (1,), device=dev) >= cfg['feat_dropout']
            xb = xb * keep
        if cfg.get('time_mask', 0) > 0:  # zero random 0.4-1.6 s spans of input frames
            B, T = xb.shape[:2]
            for _ in range(cfg['time_mask']):
                start = torch.randint(T, (B, 1), device=dev)
                length = torch.randint(5, 21, (B, 1), device=dev)
                t = torch.arange(T, device=dev)[None]
                xb = xb * ~((t >= start) & (t < start + length))[..., None, None]
        return xb

    for name, cfg in configs.items():  # CatTop on cached tap (first-1) must reproduce the cached upper taps
        if cfg.get('tune'):
            net, first = models[name], cfg['tune'].get('first', 16)
            n = min(1500, off[1] - off[0])  # conversation start: the cache had no more left context either
            x = X[:n][None]
            i = LOADED.index(first - 1)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                o = net.top(x[0, :, :, i * TAP_DIM:(i + 1) * TAP_DIM].transpose(0, 1).float())
            for t in [t for t in LOADED if t >= first] + ['final']:
                ref = (x[0, :, :, LOADED.index(t) * TAP_DIM:(LOADED.index(t) + 1) * TAP_DIM] if t != 'final'
                       else x[0, :, :, len(LOADED) * TAP_DIM:]).transpose(0, 1).float()
                cos = torch.nn.functional.cosine_similarity(o[t].float(), ref, dim=-1)
                print(f'{name}: CatTop vs cached {t}: cosine mean {cos.mean():.4f} min {cos.min():.4f}', flush=True)
            break

    # Fixed dev crops for comparable validation loss.
    g = torch.Generator(device=dev).manual_seed(1)
    dstarts = (torch.cat([torch.arange(a + WARM, b - crop, crop, device=dev) for a, b in zip(offd[:-1], offd[1:])])
               if dev_ids else None)
    stats, stop = [], threading.Event()
    threading.Thread(target=gpu_monitor, args=(stats, stop), daemon=True).start()
    history = {n: [] for n in models}
    best = {n: (float('inf'), None, 0) for n in models}
    tstep = time.time()
    for step in range(1, steps + 1):
        idx = starts[torch.randint(len(starts), (batch,), device=dev)]
        xb, yb = crops(X, Y, idx)
        xb, yb = swap_speakers(xb, yb, torch.rand(batch, device=dev) < 0.5)
        for name, net in models.items():
            net.train()
            cfg = configs[name]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = run_net(net, cfg, regularize(xb, cfg), WARM)
            target = yb
            if fine_cw is not None and cfg.get('fine_balance', 0) > 0:
                target = dict(target, fine_w=yb['fine_w'] * fine_cw[name][yb['fine']])
            if cfg.get('blur_frames', 0) > 1:  # temporal label smoothing of the floor targets
                k = cfg['blur_frames']
                kern = torch.bartlett_window(k + 2, periodic=False, device=dev)[1:-1]
                kern = (kern / kern.sum()).view(1, 1, -1)
                def blur(t):  # t [B, T, ..., 4] soft targets, smoothed along T
                    shp = t.shape
                    flat = t.movedim(1, -1).reshape(-1, 1, shp[1])
                    sm = torch.nn.functional.conv1d(torch.nn.functional.pad(flat, (k // 2, k - 1 - k // 2), mode='replicate'), kern)
                    return sm.reshape(*t.movedim(1, -1).shape).movedim(-1, 1)
                target = dict(target, floor=blur(yb['floor']), future=blur(yb['future']))
            if cfg.get('pause_weight', 0) > 0:  # up-weight frames where nobody is claiming the floor
                quiet = (yb['act'] != lb.A['CLAIM']).all(-1).float()
                boost = 1 + cfg['pause_weight'] * quiet
                target = dict(yb, floor_w=yb['floor_w'] * boost, future_w=yb['future_w'] * boost[..., None])
            loss, parts = m.loss({k: v.float() for k, v in out.items()}, target, cfg.get('loss_weights'))
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opts[name].step()
            scheds[name].step()
        if not dev_ids and step == steps:  # no dev set: keep the final weights
            for name, net in models.items():
                best[name] = (float('nan'), cat_top.trainable_state(net), step)
        if dev_ids and (step % eval_every == 0 or step == steps):
            for name, net in models.items():
                net.eval()
                tot = {}
                with torch.no_grad():
                    for i in range(0, len(dstarts), 128):
                        xb, yb = crops(Xd, Yd, dstarts[i:i + 128])
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            out = run_net(net, configs[name], xb, WARM)
                        _, parts = m.loss({k: v.float() for k, v in out.items()}, yb)
                        for k, v in parts.items():
                            tot[k] = tot.get(k, 0) + v * len(xb)
                tot = {k: v / len(dstarts) for k, v in tot.items()}
                history[name].append(dict(step=step, **tot))
                key = tot['floor'] + tot['future']
                if key < best[name][0]:
                    best[name] = (key, cat_top.trainable_state(net), step)
                print(f'step {step} {name}: ' + ' '.join(f'{k} {v:.4f}' for k, v in tot.items()), flush=True)
            util = [u for u, _ in stats[-30:]]
            print(f'  {(time.time() - tstep) / step * 1000:.0f} ms/step for {len(models)} models; '
                  f'GPU util {np.mean(util) if util else -1:.0f}% mem {max(mm for _, mm in stats) if stats else -1} MiB',
                  flush=True)
    del X, Y
    if not infer:  # pilot: timing/memory/numerics only
        stop.set()
        return {n: dict(best_step=best[n][2], best=best[n][0], ms_per_step=(time.time() - tstep) / steps * 1000,
                        peak_gib=torch.cuda.max_memory_allocated() / 2**30) for n in history}
    for name, net in models.items():  # restore the best dev checkpoint (early stopping)
        net.load_state_dict(best[name][1], strict=False)  # trainable params; frozen encoder weights unchanged
        print(f'{name}: best step {best[name][2]} (floor+future {best[name][0]:.4f})', flush=True)
    torch.cuda.empty_cache()

    probs = infer_all(models, configs, (('oto', Xd, offd, dev_ids),), dev) if dev_ids else {}
    if dev_ids:
        del Xd
    torch.cuda.empty_cache()
    Xt, offt, _ = load_split('tbdev', tb_ids, dev, cols)
    probs.update(infer_all(models, configs, (('tbdev', Xt, offt, tb_ids),), dev))
    os.makedirs(f'/work/runs/{run}', exist_ok=True)
    np.savez_compressed(f'/work/runs/{run}/probs.npz', **probs)
    for name, net in models.items():
        torch.save(dict(cfg=configs[name], state=cat_top.trainable_state(net)), f'/work/runs/{run}/{name}.pt')
    json.dump(dict(configs=configs, loaded_taps=LOADED, train_ids=train_ids, history=history, best_steps={n: b[2] for n, b in best.items()}, n_train=len(train_ids), steps=steps, batch=batch, crop=crop,
                   wall_s=time.time() - t0, gpu_util=[u for u, _ in stats]), open(f'/work/runs/{run}/train.json', 'w'))
    work.commit()
    stop.set()
    return {n: dict(best_step=best[n][2], best=best[n][0]) for n in history}


@app.local_entrypoint()
def main(run: str, configs: str, n_train: int = 32, steps: int = 1500, batch: int = 64, gpu: str = 'A100',
         extra: bool = False, seed: int = 0, no_dev: bool = False, train_from: str = '', eval_every: int = 250,
         no_infer: bool = False):
    cfgs = json.load(open(configs))
    print(train.with_options(gpu=gpu).remote(run, cfgs, n_train, steps, batch, 375, eval_every, seed, extra,
                                             not no_dev, train_from or None, not no_infer))
