"""Paired causal-GRU head runs on completed full-scale caches (mirrors run_full_scale.py).

  python run_gru.py frozen            # cache-streaming-full-v1
  python run_gru.py continued         # cache-vap-update100-full-v1
Seeds 42 and 17, 12 epochs, same dev set/scorer as the MLP heads. Resumes existing run dirs.
"""
from pathlib import Path
import json, os, subprocess, sys
ROOT = Path('/content/turn-recreation')
CACHES = {'frozen': 'cache-streaming-full-v1', 'continued': 'cache-vap-update100-full-v1'}
os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('MKL_NUM_THREADS', '4')
for kind in sys.argv[1:] or list(CACHES):
    manifest = ROOT / CACHES[kind] / 'manifest.json'
    assert len(json.loads(manifest.read_text())) == 147, f'{kind} cache incomplete'
    for seed in (42, 17):
        out = ROOT / 'runs' / f'full-{kind}-gru-seed{seed}'
        argv = ['temporal_head.py', 'train', '--epochs', '12', '--seed', str(seed),
                '--manifest', str(manifest), '--out', str(out)]
        if out.exists():
            argv += ['--resume']
        print('START', argv, flush=True)
        subprocess.run([sys.executable, *argv], cwd=ROOT, check=True)
        print('GRU_COMPLETE', kind, seed, (out / 'best.json').read_text(), flush=True)
print('GRU_ALL_COMPLETE', flush=True)
