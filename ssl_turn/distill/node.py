"""colab.py on Rin's Ray node (RTX 3090), without the root symlinks colab.py setup makes.

Data lives in $WORK (default ~/work/distill); the pipeline code reads it through SSL_TURN_WORK.
Submitted from a cloud thread with ../ray_submit.py, e.g.

    python ray_submit.py --script distill/node.py --gpus 1 --python 'PYTHONPATH=~/work/distill/pydeps ~/work/ssl-turn/venv-torch/bin/python' \
        --forward-env MODAL_TOKEN_ID MODAL_TOKEN_SECRET HF_TOKEN -- pull --feats
    ... -- train d004 configs/d004.json --steps 6000

`pull` fetches only what an otoSpeech + voice-agent run needs (no People's Speech pool, no bgbench).
`put SRC DST` copies a file back to the ssl-turn-work volume (for scoring in a cloud thread).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('WORK', os.path.expanduser('~/work/distill'))
os.environ['SSL_TURN_WORK'] = os.environ['WORK']
sys.path.insert(0, HERE)
import colab  # noqa: E402

colab.PULL = ['split.json', 'gold', 'labels/oto', 'labels/eotb', 'audio16/oto', 'audio16/tbdev', 'audio16/eotb',
              'audio16_aug/oto', f'teacher/{colab.TEACHER}', 'runs/r025_eotb5/eotb5_s2.pt']
colab.PULL_FEATS = ['feats_asr/oto', 'feats_asr/eotb']
colab.PULL_BG = []

if __name__ == '__main__':
    os.makedirs(os.environ['WORK'], exist_ok=True)
    if sys.argv[1:2] == ['put']:
        src, dst = sys.argv[2:4]
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'put', '--force', 'ssl-turn-work', src, dst], check=True)
    else:
        colab.main()
