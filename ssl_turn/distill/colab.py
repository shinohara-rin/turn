"""Run the distillation experiments on one GPU machine (e.g. a Colab A100) instead of Modal.

The Modal functions in distill.py, pool.py, ../pipeline/policy_score.py and ../../bgspeech/bench_modal.py
are plain Python under their decorators. This runner installs a stand-in `modal` module (decorators
return the function; .remote/.map/.starmap call it locally, starmap in a process pool; volume
commits are no-ops) and points the container paths at local directories with symlinks:
  /work -> $WORK (default /content/work)     the ssl-turn-work volume
  /vol  -> $BGVOL (default /content/bgbench) the turn-bgbench volume
  /root/ssl_turn, /root/bgspeech -> this checkout

    python colab.py setup
    python colab.py pull [--feats]           # prepared data from the Modal volumes (needs `modal` + token)
    python colab.py vad                      # Silero per 80 ms frame for the vad_in variant
    python colab.py train RUN configs/c001.json [--steps 10000] [--select ood] [--keep-all]
    python colab.py eot RUN s3m,fc4 [--lang en] --out tracks.npz
    python colab.py sweep RUN s3m            # TB dev through the commit policy (turnbench venv)
    python colab.py bg RUN s3m,fc4           # bgbench outputs, then pack to $BGVOL/../bg/RUN.npz
    python colab.py eotsum 'en/ssl_turn__*'  # eot-bench table (eot-harness venv)
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                       # ssl_turn/
REPO = os.path.dirname(SRC)
WORK = os.environ.get('WORK', '/content/work')
BGVOL = os.environ.get('BGVOL', '/content/bgbench')
TEACHER = 'r025s2'                                # teacher/{tag}: r025_eotb5 eotb5_s2 (latency/README.md)


# ---- a local stand-in for modal -------------------------------------------------------------------

def _call(spec, args):
    mod, name = spec
    return getattr(sys.modules[mod], name).f(*args)


class _Fn:
    """A decorated function: callable as is, with Modal's call styles run locally."""

    def __init__(self, f):
        self.f = f
        functools.update_wrapper(self, f)

    def __call__(self, *a, **k):
        return self.f(*a, **k)

    remote = local = __call__

    def with_options(self, **k):
        return self

    def spawn(self, *a, **k):
        r = self.f(*a, **k)
        return types.SimpleNamespace(get=lambda: r)

    def map(self, *its, **k):
        return self.starmap(zip(*its))

    def starmap(self, it, **k):
        items = list(it)
        procs = int(os.environ.get('LOCAL_PROCS', os.cpu_count() or 1))
        if procs <= 1 or len(items) <= 1:
            return [self.f(*a) for a in items]
        spec = (self.f.__module__, self.f.__name__)
        with mp.get_context('fork').Pool(min(procs, len(items))) as pool:
            return pool.starmap(_call, [(spec, a) for a in items])


class _Any:
    """Absorbs any builder chain (Image..., Secret..., Volume methods)."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()

    def __iter__(self):
        return iter(())


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return _Fn

    def local_entrypoint(self, *a, **k):
        return _Fn

    def __getattr__(self, name):
        return _Any()


def install_modal_stub():
    stub = types.ModuleType('modal')
    stub.App = _App
    stub.Image = _Any()
    stub.Volume = _Any()
    stub.Secret = _Any()
    stub.is_local = lambda: False
    sys.modules['modal'] = stub
    for p in (HERE, SRC, os.path.join(SRC, 'pipeline'), os.path.join(REPO, 'bgspeech'), os.path.join(REPO, 'eotbench')):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['SSL_TURN_SRC'] = SRC


def _link(target, link):
    if os.path.islink(link) or os.path.exists(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return
        raise SystemExit(f'{link} exists and is not a link to {target}')
    os.symlink(target, link)


def cmd_setup(a):
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(BGVOL, exist_ok=True)
    _link(WORK, '/work')
    _link(BGVOL, '/vol')
    os.makedirs('/root', exist_ok=True)
    _link(SRC, '/root/ssl_turn')
    _link(os.path.join(REPO, 'bgspeech'), '/root/bgspeech')
    print('linked /work, /vol, /root/ssl_turn, /root/bgspeech')


# ---- data -----------------------------------------------------------------------------------------

PULL = ['split.json', 'gold', 'labels/oto', 'labels/eotb', 'audio16/oto', 'audio16/tbdev', 'audio16/eotb',
        'audio16/pool', 'audio16_aug/oto', f'teacher/{TEACHER}', 'vad/tbdev',
        'runs/r025_eotb5/eotb5_s2.pt', 'runs/r025_eotb5/probs.npz']
PULL_FEATS = ['feats_asr/oto', 'feats_asr/eotb', 'feats_asr/pool']   # pool feature distillation only
PULL_BG = ['bench', 'musan/fma']


def _get(volume, remote, root):
    dst = os.path.join(root, remote)
    if os.path.exists(dst):
        print('have', dst, flush=True)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print('modal volume get', volume, remote, flush=True)
    subprocess.run(['modal', 'volume', 'get', volume, remote, os.path.dirname(dst) or '.'], check=True)


def cmd_pull(a):
    """Everything the student runs read, ~46 GB (+36 GB with --feats)."""
    for p in PULL + (PULL_FEATS if a.feats else []):
        _get('ssl-turn-work', p, WORK)
    for p in PULL_BG:
        _get('turn-bgbench', p, BGVOL)


def _vad_job(split, cid, aug):
    import numpy as np
    import augment
    src = f'{WORK}/audio16{"_aug" if aug else ""}/{split}/{cid}.npy'
    out = f'{WORK}/vadf/{split}{"_aug" if aug else ""}/{cid}.npy'
    if os.path.exists(out) or not os.path.exists(src):
        return
    w = np.load(src).astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + '.tmp.npy', augment.vad_frames(w if w.ndim == 2 else w[:, None]))
    os.replace(out + '.tmp.npy', out)


def cmd_vad(a):
    """Silero VAD per 80 ms frame ({WORK}/vadf) for every training source of the vad_in variant."""
    jobs = []
    for split in ('oto', 'eotb', 'pool'):
        d = f'{WORK}/audio16/{split}'
        if os.path.isdir(d):
            cids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npy') and not f.endswith('.tmp.npy'))
            jobs += [(split, c, False) for c in cids]
            if split == 'oto':
                jobs += [(split, c, True) for c in cids]
    print(len(jobs), 'files', flush=True)
    with mp.get_context('fork').Pool(os.cpu_count()) as pool:
        for i, _ in enumerate(pool.starmap(_vad_job, jobs, chunksize=4)):
            pass
    print('done')


# ---- experiments ----------------------------------------------------------------------------------

def cmd_train(a):
    import distill as ds
    cfgs = json.load(open(a.configs if os.path.isabs(a.configs) else os.path.join(HERE, a.configs)))
    extra = ds._extra() if a.extra else ()
    print(ds.train_students(a.run, cfgs, a.teacher, a.steps, a.batch, 375, a.eval_every, a.lr,
                            infer=not a.no_infer, extra_ids=extra, data_device=a.data_device, select=a.select,
                            keep_all=a.keep_all), flush=True)


def cmd_infer(a):
    import distill as ds
    print(ds.infer(a.run, a.names.split(',')))


def cmd_eot(a):
    import distill as ds
    with open(a.out, 'wb') as f:
        f.write(ds.eot_tracks(a.run, a.names.split(','), a.lang))
    print('wrote', a.out)


def cmd_sweep(a):
    import policy_score
    os.environ.setdefault('LOCAL_PROCS', str(os.cpu_count()))
    policy_score.sweep(run=a.run, models=a.models)


def cmd_bg(a):
    import bench_modal as bm
    import numpy as np
    plan = bm._plan()
    bm.student_group(plan['ids'], a.run, a.names.split(','))
    out = os.path.join(os.path.dirname(BGVOL), 'bg')
    os.makedirs(out, exist_ok=True)
    with open(f'{out}/{a.run}.npz', 'wb') as f:
        f.write(bm.pack(a.run))
    json.dump(plan, open(f'{out}/plan.json', 'w'), indent=1)
    print('wrote', f'{out}/{a.run}.npz', np.__version__)


def cmd_eotsum(a):
    """eot-bench summary of harness output dirs (run in the eot-harness venv): AUC, mean latency at
    2 / 5 / 10% false cutoffs, cutoff rate at 300 ms, pause-onset AUC at 0.1 / 0.3 s."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    rows = []
    for d in sorted(glob.glob(a.glob)):
        if not os.path.exists(f'{d}/metrics/summary.json'):
            continue
        s = json.load(open(f'{d}/metrics/summary.json'))
        t = pd.read_parquet(f'{d}/metrics/tradeoff.parquet')
        ok = t[t.mean_latency <= 0.3 + 1e-9]
        p = pd.read_parquet(f'{d}/predictions.parquet')
        onset = [roc_auc_score(q.label == 'eot', q.p_eot) for q in
                 (p[np.isclose(p.silence_dur, sd)] for sd in (0.1, 0.3))]
        op = s['operating_points']
        rows.append(dict(model=os.path.basename(d).replace('ssl_turn__', ''), auc=s['auc'],
                         l2=op['2pct']['mean_latency'] * 1000, l5=op['5pct']['mean_latency'] * 1000,
                         l10=op['10pct']['mean_latency'] * 1000,
                         c300=ok.cutoff_rate.min() * 100 if len(ok) else float('nan'),
                         onset01=onset[0], onset03=onset[1]))
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('setup')
    p = sub.add_parser('pull')
    p.add_argument('--feats', action='store_true')
    sub.add_parser('vad')
    p = sub.add_parser('train')
    p.add_argument('run')
    p.add_argument('configs')
    p.add_argument('--teacher', default=TEACHER)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--select', default='oto', choices=['oto', 'ood'])
    p.add_argument('--keep-all', action='store_true')
    p.add_argument('--no-extra', dest='extra', action='store_false',
                   help='leave out the otoSpeech conversations outside train and dev (d003 used them)')
    p.add_argument('--no-infer', action='store_true')
    p.add_argument('--data-device', default='cpu')
    p = sub.add_parser('infer')
    p.add_argument('run')
    p.add_argument('names')
    p = sub.add_parser('eot')
    p.add_argument('run')
    p.add_argument('names')
    p.add_argument('--lang', default='en')
    p.add_argument('--out', default='tracks.npz')
    p = sub.add_parser('sweep')
    p.add_argument('run')
    p.add_argument('models')
    p = sub.add_parser('bg')
    p.add_argument('run')
    p.add_argument('names')
    p = sub.add_parser('eotsum')
    p.add_argument('glob')
    a = ap.parse_args()
    if a.cmd not in ('setup', 'pull', 'eotsum'):
        install_modal_stub()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
