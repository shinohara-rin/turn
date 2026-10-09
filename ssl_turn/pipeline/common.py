"""Shared Modal definitions for the ssl_turn pipeline.

Volumes:
  turnbench-datasets (read only): pinned otoSpeech, TurnBench dev/test, staged by attempt 1.
  ssl-turn-work: this attempt's split, labels, resampled audio, features, runs.

TurnBench test audio is never opened by this pipeline.
"""
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SRC = HERE.parent  # ssl_turn/

datasets = modal.Volume.from_name('turnbench-datasets')
work = modal.Volume.from_name('ssl-turn-work', create_if_missing=True)
VOLUMES = {'/datasets': datasets, '/work': work}

# Paths default to the Modal volume mounts; ray_run.py points them at a local disk instead.
WORK = os.environ.get('SSL_TURN_WORK', '/work')
OTO = os.environ.get('SSL_TURN_OTO', '/datasets/otoearth/otoSpeech-full-duplex-turn-104h')
TB_DEV = os.environ.get('SSL_TURN_TB_DEV', '/datasets/mundo-ai/turn-benchmark-dev/data')
TURNBENCH_COMMIT = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'


def _code(image):
    return (image.add_local_dir(str(SRC), '/root/ssl_turn', ignore=['**/__pycache__'])
            .add_local_python_source('common'))


# CPU image with the pinned evaluator (its own dependency pins), for labels and scoring.
cpu_image = _code(
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git')
    .pip_install(f'git+https://github.com/SesameAILabs/turnbench@{TURNBENCH_COMMIT}', 'srt', 'scipy')
)

# GPU image: torch + transformers for the encoders and the turn model.
gpu_image = _code(
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'transformers==5.19.0', 'safetensors', 'numpy', 'scipy', 'soundfile',
                 'huggingface_hub', 'httpx')
)


def setup_path():
    import sys
    src = os.environ.get('SSL_TURN_SRC', '/root/ssl_turn')
    if src not in sys.path:
        sys.path.insert(0, src)


def gpu_monitor(log, stop):
    import subprocess
    while not stop.is_set():
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip()
            log.append(tuple(int(v) for v in q.split(',')))
        except Exception:
            pass
        stop.wait(2)
