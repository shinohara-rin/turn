"""Encoder vs head: is the remaining error in the frozen features or in the floor head?

Two tests on the same decision events, read out causally:
  1. Head fit: the trained heads' scores on their own training conversations vs oto dev and
     TurnBench dev. A head that cannot separate the hard cases even on train is limited by
     its input or capacity; one that separates them on train but not on dev is limited by data.
  2. Probes: linear and MLP probes on frozen features at the same readout frame. A probe that
     matches the head says the head adds nothing beyond the features; a large gap says the
     head (its 20 s context and floor objective) is doing real work.

Events come from the pinned gold builder's events (gold/oto single-annotator, gold/tbdev
three-annotator):
  EOT: turn ends (positive) vs holds (negative), read at boundary + 0.24 s and + 0.48 s;
       holds shorter than the readout delay are dropped (the pause is already over).
  INT: floor-taking onsets (positive) vs backchannel / non-content spans (negative), read at
       onset + 0.40 s; negative spans shorter than 0.40 s are dropped (they end before the
       decision, where the residuals show false fires do not happen).
Readout frame for time t is floor(t / 80 ms) - 1, whose features end at t (causal).
Metric: ROC AUC and FPR at TPR 0.90 per event set. Threshold-free; no commit policy.

    modal run diagnose.py --run r012_fine
"""
import json

import modal

from common import WORK, VOLUMES, gpu_image, setup_path, work

app = modal.App('ssl-turn-diagnose')
FRAME_S = 0.08
EOT_DELAYS = (0.24, 0.48)
INT_DELAY = 0.40
MEAN_FRAMES = 13  # 1.04 s trailing mean (incl. the readout frame)
CAT_SETS = {'tap7': (7,), 'tap15': (15,), 'tap23': (23,), 'tap31': (31,), 'final': ('final',),
            'head_input': (15, 23, 31, 'final')}


def readout(t):
    return int(t / FRAME_S) - 1


def make_events(split, cid, gold, fine=None, resid=None):
    """Decision events for one conversation. fine: [T, 2] FINE indices (oto) for categories;
    resid: TurnBench dev residual rows (r012 residuals.json, same order) for categories."""
    setup_path()
    import labels as lb
    import numpy as np
    bc = set(lb.FINE_GROUPS['backchannel'])
    ev = []
    e = gold['events']
    neg_i = {'eot': 0, 'int': 0}
    for s in (1, 2):
        c = s - 1
        for x in (x for x in e['eot_positive_events'] if x['speaker'] == s):
            for d in EOT_DELAYS:
                ev.append(dict(task=f'eot{int(d * 100)}', y=1, c=c, i=readout(x['time_s'] + d), cat='end'))
        for x in (x for x in e['eot_negative_spans'] if x['speaker'] == s):
            a, b = x['start'], x['end']
            r = resid['eot_neg'][neg_i['eot']] if resid else None
            neg_i['eot'] += 1
            if fine is not None:
                lo, hi = int(a / FRAME_S), int(b / FRAME_S) + 1
                other_bc = bool(np.isin(fine[lo:hi, 1 - c], list(bc)).any())
            else:
                other_bc = bool(r['other_backchannel'])
            cats = ['long_hold' if b - a >= 1.2 else 'short_hold'] + (['hold_other_bc'] if other_bc else [])
            for d in EOT_DELAYS:
                if b - a > d:
                    ev.append(dict(task=f'eot{int(d * 100)}', y=0, c=c, i=readout(a + d), cat=cats))
        for x in (x for x in e['int_positive_events'] if x['speaker'] == s):
            ev.append(dict(task='int40', y=1, c=c, i=readout(x['time_s'] + INT_DELAY), cat='int'))
        for x in (x for x in e['int_negative_spans'] if x['speaker'] == s):
            a, b = x['start'], x['end']
            r = resid['int_neg'][neg_i['int']] if resid else None
            neg_i['int'] += 1
            if b - a <= INT_DELAY:
                continue
            if fine is not None:
                lo, hi = int(a / FRAME_S), int(b / FRAME_S) + 1
                f = fine[lo:hi, c]
                f = f[f > 0]
                label = lb.FINE[np.bincount(f).argmax()] if len(f) else 'SILENT'
            else:
                label = r['label'] or 'none'
            ev.append(dict(task='int40', y=0, c=c, i=readout(a + INT_DELAY), cat=[label]))
    for x in ev:
        x.update(split=split, cid=cid)
        if isinstance(x['cat'], str):
            x['cat'] = [x['cat']]
    return ev


def gather(X, ev):
    """X [T, 2, D] numpy -> [n, 4, D] fp16: own now, other now, own 1 s mean, other 1 s mean."""
    import numpy as np
    out = np.empty((len(ev), 4, X.shape[-1]), np.float16)
    for k, x in enumerate(ev):
        i, c = min(x['i'], len(X) - 1), x['c']
        w = X[max(0, i - MEAN_FRAMES + 1):i + 1].astype(np.float32).mean(0)
        out[k] = np.stack([X[i, c], X[i, 1 - c], w[c], w[1 - c]])
    return out


def auc(y, s):
    import numpy as np
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s, np.float64)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def fpr_at(y, s, tpr=0.9):
    import numpy as np
    y, s = np.asarray(y), np.asarray(s)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float('nan')
    th = np.quantile(s[y == 1], 1 - tpr)
    return float((s[y == 0] >= th).mean())


def train_probe(Xtr, ytr, Xsel, ysel, kind, wd, dev='cuda', epochs=60, seed=0):
    """Logistic or 1-hidden-layer MLP probe; class-balanced BCE; early stopping on sel AUC."""
    import torch
    torch.manual_seed(seed)
    D = Xtr.shape[1]
    net = (torch.nn.Linear(D, 1) if kind == 'linear' else
           torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(D, 512), torch.nn.GELU(),
                               torch.nn.Dropout(0.3), torch.nn.Linear(512, 1))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    pos = ytr.float().mean()
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).clamp(max=50))
    best, best_state = -1, None
    n = len(Xtr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for a in range(0, n, 1024):
            idx = perm[a:a + 1024]
            loss = lossf(net(Xtr[idx]).squeeze(-1), ytr[idx].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                a_ = auc(ysel.cpu().numpy(), net(Xsel).squeeze(-1).float().cpu().numpy())
            if a_ > best:
                best, best_state = a_, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, best


@app.function(image=gpu_image, volumes=VOLUMES, gpu='L4', cpu=8, memory=65536, timeout=5400)
def diagnose(run='r012_fine', models=('fine1_bal1_s1', 'fine1_bal1_s2', 'nofine_s1', 'nofine_s2')):
    import os
    import sys
    import time
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    setup_path()
    sys.path.insert(0, '/root/ssl_turn/pipeline')
    import train as tr
    import score as sc
    t0 = time.time()
    dev = 'cuda'
    split = json.load(open(f'{WORK}/split.json'))['splits']
    tb_gold = json.load(open(f'{WORK}/gold/tbdev.json'))
    resid = json.load(open(f'{WORK}/runs/{run}/residuals.json'))
    mtd_ids = set(json.load(open(f'{WORK}/mtd_train_ids.json')))
    # Split TurnBench dev residual rows per conversation in the order residuals() emitted them.
    per = {cid: {'eot_neg': [], 'int_neg': []} for cid in tb_gold}
    for key in ('eot_neg', 'int_neg'):
        it = iter(resid[key])
        for cid in sorted(tb_gold, key=int):
            ev = tb_gold[cid]['events']
            n = sum(1 for s in (1, 2) for x in ev['eot_negative_spans' if key == 'eot_neg' else 'int_negative_spans']
                    if x['speaker'] == s)
            per[cid][key] = [next(it) for _ in range(n)]
            assert all(r['cid'] == cid for r in per[cid][key])

    tr.LOADED = [15, 23, 31]
    head_cols = tr.columns(tr.LOADED)
    nets, cfgs = {}, {}
    for n in models:
        ck = torch.load(f'{WORK}/runs/{run}/{n}.pt', map_location=dev)
        cfgs[n] = ck['cfg']
        nets[n] = tr.build_model(ck['cfg']).to(dev)
        nets[n].load_state_dict(ck['state'])
        nets[n].eval()

    have = {f[:-4] for f in os.listdir(f'{WORK}/feats/oto')}
    items = ([('train', c) for c in split['train'] if c in have] + [('otodev', c) for c in split['dev'] if c in have]
             + [('tbdev', c) for c in sorted(tb_gold, key=int)])

    def load(item):
        sp, cid = item
        d = 'tbdev' if sp == 'tbdev' else 'oto'
        X = np.load(f'{WORK}/feats/{d}/{cid}.npy')
        if sp == 'tbdev':
            ev = make_events(sp, cid, tb_gold[cid], resid=per[cid])
        else:
            z = np.load(f'{WORK}/labels/oto/{cid}.npz')
            ev = make_events(sp, cid, json.load(open(f'{WORK}/gold/oto/{cid}.json')), fine=z['fine'])
        mtd = None
        if sp == 'tbdev' or cid in mtd_ids:
            mtd = gather(np.load(f'{WORK}/feats_mtd/{d}/{cid}.npy', mmap_mode='r'), ev)
        return sp, cid, X, ev, gather(X, ev), mtd

    cache = f'{WORK}/runs/diag/cache_{run}.npz'
    events, feats, mfeats, heads = [], [], [], {n: {} for n in models}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        events, heads = list(z['events']), z['heads'].item()
        feats, mfeats, items = [z['F']], [z['M']], []
    def loaded():  # bounded prefetch: map() would queue every conversation's features in RAM
        with ThreadPoolExecutor(4) as pool:
            for a in range(0, len(items), 8):
                yield from pool.map(load, items[a:a + 8])

    for sp, cid, X, ev, f, mtd in loaded():
        Xg = torch.from_numpy(np.ascontiguousarray(X[..., head_cols])).to(dev)
        out = tr.infer_all(nets, cfgs, (('x', Xg, [0, len(Xg)], [cid]),), dev)
        del Xg, X
        for n in models:
            v = sc.score_variants(out[f'{n}/x/{cid}/post'].astype(np.float32),
                                  out[f'{n}/x/{cid}/silent'].astype(np.float32),
                                  out[f'{n}/x/{cid}/fine'].astype(np.float32) if f'{n}/x/{cid}/fine' in out else None)
            for key in ('eot', 'eot_q', 'int_spk', 'int_nobc'):
                if key in v:
                    tr_ = v[key]
                    heads[n].setdefault(key, []).extend(
                        float(tr_[min(x['i'], len(tr_) - 1), x['c']]) for x in ev)
        for x in ev:
            x['mtd'] = mtd is not None
        events.extend(ev)
        feats.append(f)
        if mtd is not None:
            mfeats.append(mtd)
    F = np.concatenate(feats)
    M = np.concatenate(mfeats)
    if not os.path.exists(cache):
        os.makedirs(f'{WORK}/runs/diag', exist_ok=True)
        np.savez(cache, events=np.array(events, dtype=object), heads=np.array(heads, dtype=object), F=F, M=M)
        work.commit()
    print(f'{len(events)} events from {len(items)} conversations in {time.time() - t0:.0f}s; '
          f'Cat {F.shape} MTD {M.shape}', flush=True)
    has_mtd = np.array([x['mtd'] for x in events])
    sp_arr = np.array([x['split'] for x in events])
    task_arr = np.array([x['task'] for x in events])
    y_arr = np.array([x['y'] for x in events])
    cats = [x['cat'] for x in events]

    def report(mask_fn, scores):
        """AUC / FPR@TPR0.9 per split, plus per negative category (vs all positives) on each split."""
        res = {}
        for sp in ('train', 'otodev', 'tbdev', 'mtd_train', 'mtd_sel'):
            m = mask_fn(sp)
            if m.sum() == 0:
                continue
            y, s = y_arr[m], scores[m]
            r = dict(n_pos=int(y.sum()), n_neg=int((y == 0).sum()), auc=auc(y, s), fpr90=fpr_at(y, s))
            cat_r = {}
            idx = np.where(m)[0]
            pos = idx[y_arr[idx] == 1]
            for cname in sorted({c for i in idx if y_arr[i] == 0 for c in cats[i]}):
                neg = np.array([i for i in idx if y_arr[i] == 0 and cname in cats[i]])
                if len(neg) >= 20:
                    yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                    ss = np.r_[scores[pos], scores[neg]]
                    cat_r[cname] = dict(n=len(neg), auc=auc(yy, ss), fpr90=fpr_at(yy, ss))
            r['cats'] = cat_r
            res[sp] = r
        return res

    # MTD comparison split: 19 train / 4 selection conversations among the 23 with MTD features.
    mtd_sorted = sorted(mtd_ids)
    mtd_sel_ids = set(mtd_sorted[::6])  # 4 conversations
    cid_arr = np.array([x['cid'] for x in events])

    def split_mask(sp, task):
        base = task_arr == task
        if sp == 'mtd_train':
            return base & (sp_arr == 'train') & has_mtd & ~np.isin(cid_arr, list(mtd_sel_ids))
        if sp == 'mtd_sel':
            return base & (sp_arr == 'train') & np.isin(cid_arr, list(mtd_sel_ids))
        return base & (sp_arr == sp)

    tasks = ['eot24', 'eot48', 'int40']
    head_key = {'eot24': ('eot_q', 'eot'), 'eot48': ('eot_q', 'eot'), 'int40': ('int_nobc', 'int_spk')}
    results = dict(head={}, probe={}, mtd={}, counts={})
    for task in tasks:
        results['counts'][task] = {sp: int(split_mask(sp, task).sum()) for sp in ('train', 'otodev', 'tbdev')}
        for n in models:
            for key in head_key[task]:
                if key in heads[n]:
                    s = np.array(heads[n][key])
                    results['head'][f'{task}/{n}/{key}'] = report(lambda sp: split_mask(sp, task), s)

    tap_pos = {7: 0, 15: 1, 23: 2, 31: 3}

    def cols(spec):
        c = []
        for t in spec:
            c.append(np.arange(4 * 1280, 4 * 1280 + 768) if t == 'final' else
                     np.arange(tap_pos[t] * 1280, (tap_pos[t] + 1) * 1280))
        return np.concatenate(c)

    def run_probes(feat, y, rows_train, rows_sel, tag):
        """feat/y are rows of one task's events; rows_* index into them. Standardize on train
        rows; grid over kind x wd; pick by sel AUC; return scores for every row."""
        Xall = torch.from_numpy(feat).to(dev).float()  # [n, 4*D] already flattened
        mu, sd = Xall[rows_train].mean(0), Xall[rows_train].std(0) + 1e-3
        Xall = (Xall - mu) / sd
        yall = torch.from_numpy(y).to(dev)
        out = {}
        for kind in ('linear', 'mlp'):
            best = (-1, None, None)
            for wd in ((1e-2, 1e-1, 1.0) if kind == 'linear' else (1e-2, 1e-1)):
                net, a = train_probe(Xall[rows_train], yall[rows_train], Xall[rows_sel], yall[rows_sel], kind, wd)
                if a > best[0]:
                    best = (a, wd, net)
            with torch.no_grad():
                s = torch.cat([best[2](Xall[i:i + 4096]).squeeze(-1) for i in range(0, len(Xall), 4096)])
            out[kind] = (best[1], s.float().cpu().numpy())
            print(f'  {tag} {kind}: wd {best[1]} sel AUC {best[0]:.3f}', flush=True)
        del Xall
        torch.cuda.empty_cache()
        return out

    for task in tasks:
        m_task = task_arr == task
        idx = np.where(m_task)[0]
        local = {sp: np.where(split_mask(sp, task)[idx])[0] for sp in ('train', 'otodev', 'tbdev', 'mtd_train', 'mtd_sel')}
        for name, spec in CAT_SETS.items():
            feat = F[idx][..., cols(spec)].reshape(len(idx), -1)
            pr = run_probes(feat, y_arr[idx], torch.from_numpy(local['train']).to(dev), torch.from_numpy(local['otodev']).to(dev),
                            f'{task} cat:{name}')
            for kind, (wd, s) in pr.items():
                full = np.full(len(events), np.nan)
                full[idx] = s
                results['probe'][f'{task}/cat:{name}/{kind}'] = dict(
                    wd=wd, **report(lambda sp: split_mask(sp, task) & ~np.isnan(full), full))
        # Equal-data backbone comparison on the 23 MTD conversations (19 train / 4 selection).
        midx = np.where(m_task & has_mtd)[0]
        mpos = {i: k for k, i in enumerate(np.where(has_mtd)[0])}
        mlocal = {sp: np.where(split_mask(sp, task)[midx])[0] for sp in ('mtd_train', 'mtd_sel', 'tbdev')}
        for name, feat in (('cat:head_input', F[midx][..., cols(CAT_SETS['head_input'])].reshape(len(midx), -1)),
                           ('mtd', M[[mpos[i] for i in midx]].reshape(len(midx), -1))):
            pr = run_probes(feat, y_arr[midx], torch.from_numpy(mlocal['mtd_train']).to(dev),
                            torch.from_numpy(mlocal['mtd_sel']).to(dev), f'{task} 23conv {name}')
            for kind, (wd, s) in pr.items():
                full = np.full(len(events), np.nan)
                full[midx] = s
                results['mtd'][f'{task}/{name}/{kind}'] = dict(
                    wd=wd, **report(lambda sp: split_mask(sp, task) & ~np.isnan(full), full))

    os.makedirs(f'{WORK}/runs/diag', exist_ok=True)
    json.dump(results, open(f'{WORK}/runs/diag/diag.json', 'w'))
    work.commit()
    results['wall_s'] = time.time() - t0
    return results


def summary(res):
    lines = []
    for task in ('eot24', 'eot48', 'int40'):
        lines.append(f'\n== {task}  events {res["counts"][task]}')
        lines.append(f'{"system":42s} {"train":>13s} {"otodev":>13s} {"tbdev":>13s}   (AUC / FPR@TPR.9)')
        for group in ('head', 'probe', 'mtd'):
            for k, v in res[group].items():
                if not k.startswith(task + '/'):
                    continue
                cells = []
                for sp in (('mtd_train', 'mtd_sel', 'tbdev') if group == 'mtd' else ('train', 'otodev', 'tbdev')):
                    r = v.get(sp)
                    cells.append(f'{r["auc"]:.3f}/{r["fpr90"]:.3f}' if r else '-')
                lines.append(f'{group + ":" + k[len(task) + 1:]:42s} ' + ' '.join(f'{c:>13s}' for c in cells))
    return '\n'.join(lines)


@app.local_entrypoint()
def main(run: str = 'r012_fine'):
    res = diagnose.remote(run)
    print(summary(res))
    print(f'wall {res["wall_s"]:.0f}s')
