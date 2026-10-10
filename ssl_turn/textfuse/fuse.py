"""Does transcript content help EOT beyond the r012 audio model? Discrimination at segment ends
(gold EOT ends vs holds) and frame-level fusion scored with the pinned TurnBench scorer,
against a shuffled-text control."""
import json, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from turnbench.gold import AnchorEvent, Interval
from turnbench.score import TaskScore, merge, score_task
from turnbench.sweep import commit_events
S = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'turn-detector'
FPS = 12.5
G = json.load(open(f'{S}/data/dev-gold.json'))['conversations']
E = json.load(open(f'{S}/sem/eou_{TAG}.json'))
H = np.load(f'{S}/sem/hid_{TAG}.npy').astype(np.float32)
Z = np.load(f'{S}/r012/probs.npz')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

def eot_q(m, cid):
    post = Z[f'{m}/tbdev/{cid}/post'].astype(np.float32); sil = Z[f'{m}/tbdev/{cid}/silent'].astype(np.float32)
    now = post[:, 0]
    rel = np.stack([now[:, 2] + now[:, 1], now[:, 2] + now[:, 0]], 1)
    return np.clip(rel * sil, 0, 1)

# --- items: segment ends that are gold EOT ends (1) or hold starts (0)
by = {}
for i, e in enumerate(E):
    by.setdefault((e['cid'], e['spk']), []).append(i)
rows = []
for cid, c in G.items():
    for s in (1, 2):
        idx = by.get((cid, s), [])
        ends = np.array([E[i]['end'] for i in idx])
        if not len(ends): continue
        for ev in c['eot_positive_events']:
            if ev['speaker'] != s: continue
            j = np.argmin(abs(ends - ev['time_s']))
            if abs(ends[j] - ev['time_s']) < 0.3: rows.append((idx[j], 1, None))
        for sp in c['eot_negative_spans']:
            if sp['speaker'] != s: continue
            j = np.argmin(abs(ends - sp['start']))
            if abs(ends[j] - sp['start']) < 0.3: rows.append((idx[j], 0, sp['end'] - sp['start']))
ii = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); dur = np.array([r[2] or np.nan for r in rows])
grp = np.array([E[i]['cid'] for i in ii])
pt = np.array([E[i]['p'] for i in ii])
print(f'items {len(y)}: ends {y.sum()} holds {(1 - y).sum()}')
print(f'text-only AUC (zero-shot P(end)): {roc_auc_score(y, pt):.3f}')
# learned probe on hidden states, out-of-fold by conversation
Xh = H[ii]; oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(Xh, y, grp):
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Xh[tr], y[tr]); oof[te] = clf.predict_proba(Xh[te])[:, 1]
print(f'text probe AUC (5-fold by conv): {roc_auc_score(y, oof):.3f}')
# audio: max eot_q over the first 1 s of silence after the segment end (or less if the hold is shorter)
for m in ('fine1_bal1_s1', 'fine1_bal1_s2'):
    tr_ = {}
    aud = []
    for k, i in enumerate(ii):
        e = E[i]; key = e['cid']
        if key not in tr_: tr_[key] = eot_q(m, key)
        L = min(1.0, dur[k]) if y[k] == 0 else 1.0
        a, b = int(e['end'] * FPS), int((e['end'] + L) * FPS) + 1
        seg = tr_[key][a:max(b, a + 1), e['spk'] - 1]
        aud.append(seg.max() if len(seg) else tr_[key][-1, e['spk'] - 1])
    aud = np.array(aud)
    res = {}
    for name, feats in (('audio', [lg(aud)]), ('audio+text0', [lg(aud), lg(pt)]), ('audio+probe', [lg(aud), lg(oof)]),
                        ('audio+shuf', [lg(aud), lg(np.random.default_rng(0).permutation(oof))])):
        X = np.stack(feats, 1); o = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            o[te] = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        res[name] = round(roc_auc_score(y, o), 4)
    print(m, res)
    long = ~np.isnan(dur) & (dur > 0.6) | (y == 1)
    print('  long holds (>0.6 s) vs ends: audio %.3f text0 %.3f probe %.3f' % (
        roc_auc_score(y[long], aud[long]), roc_auc_score(y[long], pt[long]), roc_auc_score(y[long], oof[long])))
    hi = aud > np.quantile(aud[y == 0], 0.75)
    print('  hard subset (audio above 75th pct of holds): n=%d holds=%d text0 AUC %.3f probe AUC %.3f' % (
        hi.sum(), (y[hi] == 0).sum(), roc_auc_score(y[hi], pt[hi]), roc_auc_score(y[hi], oof[hi])))
np.save(f'{S}/sem/probe_oof_{TAG}.npy', np.stack([ii, oof]))

# --- frame-level fusion with the pinned scorer
def golds(cid):
    e = G[cid]
    return ([AnchorEvent(**x) for x in e['eot_positive_events']], [Interval(**x) for x in e['eot_negative_spans']],
            [Interval(**x) for x in e['eot_excluded']])

def commit(p, theta, refr=0.5, rc=1.0):
    times = commit_events(p, FPS, theta, refractory_s=refr)
    above = np.asarray(p) > theta; extra, k = [], int(round(rc * FPS))
    for t in times:
        i = int(round(t * FPS)) - 1; j = i + k
        if j < len(above) and above[i:j + 1].all(): extra.append((j + 1) / FPS)
    return sorted(set(times) | set(extra))

def sweep(probs):
    out = []
    for th in np.round(np.arange(0.30, 0.995, 0.01), 3):
        tot = TaskScore()
        for cid, p in probs.items():
            merge(tot, score_task(*golds(cid)[:2], {s + 1: commit(p[:, s], th) for s in (0, 1)}, golds(cid)[2]))
        out.append((th, tot.recall, tot.fp_rate, tot.latency().p50))
    return out

def fp_at(rows, r):
    ok = [x for x in rows if x[1] >= r]
    return min(ok, key=lambda x: x[2]) if ok else None

def text_tracks(score, lat=0.3):
    """Per-frame text logit offset [T, 2] (0 where no text query is live)."""
    tr = {}
    for cid in G:
        T = len(Z[f'fine1_bal1_s1/tbdev/{cid}/silent'])
        off = np.zeros((T, 2), np.float32)
        for s in (1, 2):
            idx = sorted(by.get((cid, s), []), key=lambda i: E[i]['end'])
            for n, i in enumerate(idx):
                nxt = min([E[j]['start'] for j in idx if E[j]['start'] > E[i]['end']] + [1e9])
                lo = int(np.ceil((E[i]['end'] + lat) * FPS)); hi = int(min(nxt, 1e6) * FPS)
                off[lo:hi, s - 1] = score[i]
        tr[cid] = off
    return tr

def job(args):
    m, name, w = args
    global Z
    Z = np.load(f'{S}/r012/probs.npz')
    base = {cid: eot_q(m, cid) for cid in G}
    off = TT[name]
    probs = {cid: 1 / (1 + np.exp(-(lg(base[cid]) + w * off[cid]))) for cid in G}
    rows = sweep(probs)
    return (m, name, w, fp_at(rows, 0.92), fp_at(rows, 0.94))

if __name__ == '__main__':
    allp = np.array([e['p'] for e in E])
    zs = lg(allp) - np.median(lg(allp))
    # probe score for every segment end: train on all labelled items except the conversation itself
    allprobe = np.zeros(len(E))
    cids = np.array([e['cid'] for e in E])
    for tr, te in GroupKFold(5).split(H, groups=cids):
        trc = set(cids[tr]); m_ = np.isin(grp, list(trc))
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(Xh[m_], y[m_])
        allprobe[te] = clf.predict_proba(H[te])[:, 1]
    zp = lg(allprobe) - np.median(lg(allprobe))
    rng = np.random.default_rng(0)
    TT = {'zeroshot': text_tracks(zs), 'zeroshot_shuf': text_tracks(rng.permutation(zs)),
          'probe': text_tracks(zp), 'probe_shuf': text_tracks(rng.permutation(zp))}
    jobs = [(m, n, w) for m in ('fine1_bal1_s1', 'fine1_bal1_s2') for n in TT for w in (0.0, 0.25, 0.5, 1.0)
            if not (w == 0 and n != 'zeroshot')]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(job, jobs):
            print(r, flush=True)
