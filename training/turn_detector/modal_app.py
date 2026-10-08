"""Run the turn-detector pipeline on Modal with the same layout as Colab.

The scripts keep their historical /content paths. The image provides /content as a
plain directory whose heavy parts are symlinks into persistent Modal Volumes:

  /content/turn-recreation, /content/hf, /content/turnbench -> volume `turn-detector-workspace`
  /content/drive/MyDrive/turn-detector-recreation/datasets  -> volume `turnbench-datasets` (see modal_stage.py)
  /content/drive/MyDrive/turn-detector-recreation/runs      -> volume `turn-detector-workspace` (archive/)

Usage (from this directory; HF_TOKEN must be set locally, it is forwarded as an ephemeral Secret):
  modal run modal_app.py --cmd "python -m unittest test_heads"             # CPU
  TD_MODAL_GPU=A10G modal run modal_app.py --gpu --cmd "python cache_batch.py --batch-size 4 --output-dir /content/turn-recreation/cache-streaming-v2"
  modal run --detach modal_app.py --gpu --cmd "python continue_vap.py ..."  # long jobs

Commands run in /content/turn-recreation, so README stage commands work unchanged.
Code from this checkout is synced into the workspace volume on every run; run
artifacts (splits, caches, checkpoints) are never deleted.
"""
import os, shlex, subprocess, sys, threading
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
GPU = os.environ.get('TD_MODAL_GPU', 'L4')
WORK = '/vol/workspace'
DATA = '/vol/datasets'
LAYOUT = '/content/drive/MyDrive/turn-detector-recreation'

app = modal.App('turn-detector-training')
workspace = modal.Volume.from_name('turn-detector-workspace', create_if_missing=True)
datasets = modal.Volume.from_name('turnbench-datasets', create_if_missing=True)
if modal.is_local():
    secrets = [modal.Secret.from_local_environ(['HF_TOKEN'])]
else:
    secrets = []

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'ffmpeg', 'libsndfile1', 'build-essential')
    .pip_install_from_requirements(str(HERE / 'requirements.txt'))
    .run_commands(
        f'mkdir -p {LAYOUT} && ln -sfn {WORK}/turn-recreation /content/turn-recreation'
        f' && ln -sfn {WORK}/hf /content/hf && ln -sfn {WORK}/turnbench /content/turnbench'
        f' && ln -sfn {DATA} {LAYOUT}/datasets && ln -sfn {WORK}/archive {LAYOUT}/runs'
    )
    .env({'HF_HOME': '/content/hf', 'PYTHONUNBUFFERED': '1'})
    .add_local_dir(str(HERE), '/opt/turn_detector', ignore=['webgpu_playground/**', '__pycache__/**', '*.pyc'])
)
volumes = {WORK: workspace, DATA: datasets}


def _bootstrap():
    """Sync code into the workspace volume and ensure the pinned evaluator exists."""
    for sub in ('turn-recreation', 'hf', 'archive'):
        Path(WORK, sub).mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-r', '/opt/turn_detector/.', f'{WORK}/turn-recreation/'], check=True)
    subprocess.run([sys.executable, '/opt/turn_detector/setup_remote.py'], check=True)


def _run(cmd):
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(60):
            workspace.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    _bootstrap()
    try:
        code = subprocess.run(cmd, shell=True, cwd='/content/turn-recreation').returncode
    finally:
        stop.set()
        workspace.commit()
    if code:
        raise RuntimeError(f'command exited with {code}')


@app.function(image=image, volumes=volumes, secrets=secrets, cpu=4, memory=16384, timeout=86400)
def run_cpu(cmd: str):
    _run(cmd)


@app.function(image=image, volumes=volumes, secrets=secrets, gpu=GPU, timeout=86400)
def run_gpu(cmd: str):
    _run(cmd)


@app.local_entrypoint()
def main(cmd: str, gpu: bool = False):
    print(f'[modal_app] {"GPU " + GPU if gpu else "CPU"}: {cmd}', flush=True)
    (run_gpu if gpu else run_cpu).remote(cmd)
