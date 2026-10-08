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


def load_split(split, cids, device):
    import numpy as np
    import torch
    feats, offsets, lab = [], [0], []
    for cid in cids:
        f = np.load(f'/work/feats/{split}/{cid}.npy', mmap_mode='r')
        T = len(f)
        if split == 'oto':
            z = np.load(f'/work/labels/oto/{cid}.npz')
            T = min(T, len(z['floor']))
            lab.append({k: z[k][:T] for k in ('floor', 'floor_w', 'act', 'act_w', 'future', 'future_w', 'activity')})
        feats.append(torch.from_numpy(np.ascontiguousarray(f[:T])).to(device))
        offsets.append(offsets[-1] + T)
    return torch.cat(feats), offsets, lab


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
        idx = [TAPS.index(t) for t in cfg['taps']]
        all_taps = x[..., :len(TAPS) * TAP_DIM].unflatten(-1, (len(TAPS), TAP_DIM))
        taps = all_taps[..., idx, :]
    final = x[..., len(TAPS) * TAP_DIM:] if cfg.get('final', True) else None
    return taps, final


def build_model(cfg):
    setup_path()
    import model as m
    return m.TurnModel(tap_layers=len(cfg.get('taps') or []), tap_dim=TAP_DIM,
                       final_dim=FINAL_DIM if cfg.get('final', True) else 0,
                       dim=cfg.get('dim', 256), heads=cfg.get('heads', 4), layers=cfg.get('layers', 4),
                       window_s=cfg.get('window_s', 20.0), dropout=cfg.get('dropout', 0.1))


@app.function(image=gpu_image, volumes=VOLUMES, gpu='A100', cpu=8, memory=65536, timeout=5400)
def train(run, configs, n_train=32, steps=1500, batch=64, crop=375, eval_every=250, seed=0):
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
    dev_ids = [c for c in split['dev'] if c in have]
    tb_ids = sorted(f[:-4] for f in os.listdir('/work/feats/tbdev'))
    t0 = time.time()
    X, off, lab = load_split('oto', train_ids, dev)
    Y = stack_labels(lab, dev)
    Xd, offd, labd = load_split('oto', dev_ids, dev)
    Yd = stack_labels(labd, dev)
    print(f'loaded {len(train_ids)} train ({len(X)} frames) / {len(dev_ids)} dev in {time.time() - t0:.0f}s; '
          f'VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)

    # Valid crop starts: within one conversation.
    starts = torch.cat([torch.arange(a, b - crop, device=dev) for a, b in zip(off[:-1], off[1:]) if b - a > crop])
    models, opts, scheds = {}, {}, {}
    for name, cfg in configs.items():
        net = build_model(cfg).to(dev)
        models[name] = net
        opts[name] = torch.optim.AdamW(net.parameters(), lr=cfg.get('lr', 3e-4), weight_decay=cfg.get('wd', 0.05))
        scheds[name] = torch.optim.lr_scheduler.OneCycleLR(opts[name], max_lr=cfg.get('lr', 3e-4), total_steps=steps,
                                                            pct_start=0.1)
        print(name, cfg, f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    def crops(Xs, Ys, idx):
        span = idx[:, None] + torch.arange(crop, device=dev)
        b = {k: v[span] for k, v in Ys.items()}
        return Xs[span], b

    SWAP = torch.tensor([1, 0, 2, 3], device=dev)  # HELD_0 <-> HELD_1
    VAPN = len(m.VAP_BINS)

    def swap_speakers(xb, yb, mask):
        """Channel-swap augmentation on rows in `mask`: features, acts, floor holders, VAP bits."""
        xb = torch.where(mask[:, None, None, None], xb.flip(2), xb)
        out = dict(yb)
        for k in ('act', 'act_w', 'activity'):
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

    # Fixed dev crops for comparable validation loss.
    g = torch.Generator(device=dev).manual_seed(1)
    dstarts = torch.cat([torch.arange(a, b - crop, crop, device=dev) for a, b in zip(offd[:-1], offd[1:])])
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
            taps, final = select_inputs(regularize(xb, cfg), cfg)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(taps, final)
            target = yb
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
        if step % eval_every == 0 or step == steps:
            for name, net in models.items():
                net.eval()
                tot = {}
                with torch.no_grad():
                    for i in range(0, len(dstarts), 128):
                        xb, yb = crops(Xd, Yd, dstarts[i:i + 128])
                        taps, final = select_inputs(xb, configs[name])
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            out = net(taps, final)
                        _, parts = m.loss({k: v.float() for k, v in out.items()}, yb)
                        for k, v in parts.items():
                            tot[k] = tot.get(k, 0) + v * len(xb)
                tot = {k: v / len(dstarts) for k, v in tot.items()}
                history[name].append(dict(step=step, **tot))
                key = tot['floor'] + tot['future']
                if key < best[name][0]:
                    best[name] = (key, {k: v.detach().clone() for k, v in net.state_dict().items()}, step)
                print(f'step {step} {name}: ' + ' '.join(f'{k} {v:.4f}' for k, v in tot.items()), flush=True)
            util = [u for u, _ in stats[-30:]]
            print(f'  {(time.time() - tstep) / step * 1000:.0f} ms/step for {len(models)} models; '
                  f'GPU util {np.mean(util) if util else -1:.0f}% mem {max(mm for _, mm in stats) if stats else -1} MiB',
                  flush=True)
    del X, Y
    for name, net in models.items():  # restore the best dev checkpoint (early stopping)
        net.load_state_dict(best[name][1])
        print(f'{name}: best step {best[name][2]} (floor+future {best[name][0]:.4f})', flush=True)
    torch.cuda.empty_cache()

    # Exact chunked causal inference: each chunk carries enough left context to cover the
    # stacked attention windows (layers x window frames).
    probs = {}
    Xt, offt, _ = load_split('tbdev', tb_ids, dev)
    for name, net in models.items():
        net.eval()
        cfg = configs[name]
        ctx = cfg.get('layers', 4) * int(round(cfg.get('window_s', 20.0) / 0.08))
        for split, Xs, offs, ids in (('oto', Xd, offd, dev_ids), ('tbdev', Xt, offt, tb_ids)):
            for cid, a, b in zip(ids, offs[:-1], offs[1:]):
                outs = []
                with torch.no_grad():
                    for s in range(a, b, 1000):
                        lo = max(a, s - ctx)
                        taps, final = select_inputs(Xs[lo:min(b, s + 1000)][None], cfg)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            o = net(taps, final)
                        outs.append({k: v[0, s - lo:].float() for k, v in o.items() if k in ('floor', 'future')})
                floor = torch.cat([o['floor'] for o in outs]).softmax(-1)
                future = torch.cat([o['future'] for o in outs]).softmax(-1)
                # Posteriors only; score.py derives EOT/INT score variants from them.
                post = torch.cat([floor[:, None], future], 1)  # [T, 1 + H, 4]
                probs[f'{name}/{split}/{cid}/post'] = post.cpu().numpy().astype(np.float16)
    os.makedirs(f'/work/runs/{run}', exist_ok=True)
    np.savez_compressed(f'/work/runs/{run}/probs.npz', **probs)
    for name, net in models.items():
        torch.save(dict(cfg=configs[name], state=net.state_dict()), f'/work/runs/{run}/{name}.pt')
    json.dump(dict(configs=configs, history=history, best_steps={n: b[2] for n, b in best.items()}, n_train=len(train_ids), steps=steps, batch=batch, crop=crop,
                   wall_s=time.time() - t0, gpu_util=[u for u, _ in stats]), open(f'/work/runs/{run}/train.json', 'w'))
    work.commit()
    stop.set()
    return {n: dict(best_step=best[n][2], best=best[n][0]) for n in history}


@app.local_entrypoint()
def main(run: str, configs: str, n_train: int = 32, steps: int = 1500, batch: int = 64, gpu: str = 'A100'):
    cfgs = json.load(open(configs))
    print(train.with_options(gpu=gpu).remote(run, cfgs, n_train, steps, batch))
