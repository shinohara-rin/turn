"""Run the Modal pipeline's functions on a plain machine (e.g. a Ray job on a local GPU box).

Every pipeline function is an ordinary Python function under its Modal decorator. This
runner installs a stand-in `modal` module whose decorators return the function unchanged
and whose volume commits are no-ops, points the paths in common.py at local directories,
and calls the functions directly:

    python ray_run.py prep                       # labels + 24 kHz audio + gold (turnbench env)
    python ray_run.py encode_mtd [--train N]     # MTD features: oto dev, TB dev, then train (GPU env)
    python ray_run.py train RUN CONFIGS.json [--n-train N] [--steps S]   # GPU env
    python ray_run.py score RUN [--variants a,b] [--refractories 0.5] [--recommits ,1.0]  # turnbench env

Paths (environment, with defaults for Rin's Ray node):
    SSL_TURN_WORK    ~/work/ssl-turn (plays the role of the /work volume)
    SSL_TURN_OTO     ~/data/hf/otoearth/otoSpeech-full-duplex-turn-104h
    SSL_TURN_TB_DEV  ~/data/hf/mundo-ai/turn-benchmark-dev/data
TurnBench test audio is never opened.
"""
import argparse
import json
import os
import shutil
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
home = os.path.expanduser('~')
os.environ.setdefault('SSL_TURN_WORK', f'{home}/work/ssl-turn')
os.environ.setdefault('SSL_TURN_OTO', f'{home}/data/hf/otoearth/otoSpeech-full-duplex-turn-104h')
os.environ.setdefault('SSL_TURN_TB_DEV', f'{home}/data/hf/mundo-ai/turn-benchmark-dev/data')
os.environ['SSL_TURN_SRC'] = HERE
for p in (HERE, os.path.join(HERE, 'pipeline')):
    if p not in sys.path:
        sys.path.insert(0, p)


class _Stub:
    """Absorbs Modal's builder calls; used as a decorator it returns the function unchanged."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return _Stub()


sys.modules['modal'] = types.SimpleNamespace(App=_Stub, Image=_Stub(), Volume=_Stub(), Secret=_Stub())
WORK = os.environ['SSL_TURN_WORK']
GPU_DEPS = ['transformers==5.19.0', 'safetensors', 'numpy', 'scipy', 'soundfile', 'huggingface_hub', 'httpx']


def gpu_deps():
    """Install the GPU env's packages once into a persistent target dir (see ray_submit.ENVS)."""
    import hashlib
    import subprocess
    import importlib.util
    # An interpreter that already has torch (ray_submit --python) skips the big download. Check
    # without importing: torch imported before numpy is on sys.path disables its numpy bridge.
    deps = GPU_DEPS if importlib.util.find_spec('torch') else ['torch'] + GPU_DEPS
    target = f'{WORK}/pydeps-' + hashlib.sha1(' '.join(deps).encode()).hexdigest()[:10]
    if not os.path.exists(f'{target}/.done'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', target, *deps], check=True)
        open(f'{target}/.done', 'w').close()
    sys.path.insert(0, target)


def present_oto(cids):
    oto = os.environ['SSL_TURN_OTO']
    need = ('speaker_1_audio.wav', 'speaker_2_audio.wav', 'speaker_1_annotation_a.srt', 'speaker_2_annotation_a.srt')
    return [c for c in cids if all(os.path.exists(f'{oto}/{c}/{f}') for f in need)]


def _oto_item(cid):
    import prep
    try:
        r = prep.oto_item(cid)
        prep.add_fine(cid)
        return r
    except Exception as e:  # a half-downloaded conversation: retried on the next prep
        return cid, f'error {e!r}'


def cmd_prep(a):
    """Labels, audio and gold for the oto train/dev split and TurnBench dev (idempotent)."""
    from multiprocessing import Pool
    import prep
    import score
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(f'{WORK}/split.json'):
        shutil.copy(os.path.join(HERE, 'split.json'), f'{WORK}/split.json')  # shipped with the job
    split = json.load(open(f'{WORK}/split.json'))['splits']
    cids = split['dev'] + split['train']
    ready = present_oto(cids)
    print(f'oto: {len(ready)}/{len(cids)} conversations downloaded', flush=True)
    procs = max(1, min(6, (os.cpu_count() or 2) // 2))  # ~2 GB each; the node is shared
    with Pool(procs) as pool:
        for r in pool.imap_unordered(_oto_item, ready):
            print('oto', r, flush=True)
    print('tbdev gold:', score.cache_tbdev_gold(), flush=True)
    tb = sorted(json.load(open(f'{WORK}/gold/tbdev.json')), key=int)
    todo = [c for c in tb if not os.path.exists(f'{WORK}/audio/tbdev/{c}.npy')]
    for i in range(0, len(todo), 8):  # one process: each turnbench dataset load holds ~25 GB of parquet
        print('tbdev', prep.tbdev_audio(todo[i:i + 8]), flush=True)


def cmd_encode_mtd(a):
    """MTD trailing-window features: oto dev and TB dev first, then train in split order."""
    gpu_deps()
    import encode_mtd
    print(encode_mtd.fetch_mtd(), flush=True)
    split = json.load(open(f'{WORK}/split.json'))['splits']
    tb = sorted(json.load(open(f'{WORK}/gold/tbdev.json')), key=int)
    order = ([('oto', c) for c in split['dev']] + [('tbdev', c) for c in tb]
             + [('oto', c) for c in split['train'][:a.train]])
    items = [(sp, c) for sp, c in order if os.path.exists(f'{WORK}/audio/{sp}/{c}.npy')
             and not os.path.exists(f'{WORK}/feats_mtd/{sp}/{c}.npy')]
    print(f'{len(items)} conversations to encode', flush=True)
    for i in range(0, len(items), 4):  # small batches so progress is visible and resumable
        print(encode_mtd.encode(items[i:i + 4], a.step, a.batch), flush=True)


def cmd_train(a):
    gpu_deps()
    import train
    configs = json.load(open(a.configs if os.path.isabs(a.configs) else os.path.join(HERE, 'pipeline', a.configs)))
    print(train.train(a.run, configs, a.n_train, a.steps, a.batch, 375, a.eval_every, a.seed,
                      feats_on=a.feats_on), flush=True)


def cmd_score(a):
    import score
    rc = [float(x) if x else None for x in a.recommits.split(',')]
    report = score.score_run(a.run, a.variants.split(',') if a.variants else None,
                             [float(x) for x in a.refractories.split(',')], None, rc)
    for e in report:
        print(f"{e['model']:>14} {e['task']:<8} oto[{score.fmt(e['oto'])}]  "
              f"tbdev@oto-θ[{score.fmt(e.get('tbdev_at_oto_theta'))}]  tbdev-swept[{score.fmt(e.get('tbdev_swept'))}]",
              flush=True)


def cmd_fpat(a):
    """FP (and p50) at fixed recall on TB dev from a run's saved score sweep."""
    import glob
    import score
    targets = [float(x) for x in a.recalls.split(',')]
    for f in sorted(glob.glob(f'{WORK}/runs/{a.run}/score*.json')):
        rows = json.load(open(f))['rows']
        for k in sorted(rows):
            model, task, split = k.split('|')
            if split == 'tbdev':
                print(f'{model:>14} {task:<22}', '  '.join(f'R{t}: {score._fp_at(rows[k], t)}' for t in targets),
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prep')
    e = sub.add_parser('encode_mtd')
    e.add_argument('--train', type=int, default=131)
    e.add_argument('--step', type=int, default=2)
    e.add_argument('--batch', type=int, default=48)
    t = sub.add_parser('train')
    t.add_argument('run')
    t.add_argument('configs')
    t.add_argument('--n-train', type=int, default=131)
    t.add_argument('--steps', type=int, default=1000)
    t.add_argument('--batch', type=int, default=64)
    t.add_argument('--eval-every', type=int, default=250)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--feats-on', default='cpu')
    s = sub.add_parser('score')
    s.add_argument('run')
    s.add_argument('--variants', default='')
    s.add_argument('--refractories', default='2.0')
    s.add_argument('--recommits', default='')
    f = sub.add_parser('fpat')
    f.add_argument('run')
    f.add_argument('--recalls', default='0.92,0.95,0.97')
    a = ap.parse_args()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
