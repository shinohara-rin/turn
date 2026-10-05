"""Score the official TurnBench-fine-tuned VAP (oto ckpt) on the TurnBench dev set on a Modal GPU.

Prereqs (once, on a machine where the Modal CLI can connect):
    pip install modal && modal token set ...           # or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars
    modal secret create huggingface HF_TOKEN=hf_...     # token with access to mundo-ai/turn-benchmark-dev
Run:
    modal run modal_dev.py                    # oto checkpoint (official baseline)
    modal run modal_dev.py --ckpt pretrained  # original Switchboard VAP
Outputs land in ./results-modal/ next to this file (probs, predictions-dev.json, score.txt).
"""
from pathlib import Path

import modal

TURNBENCH_SHA = "38a6f874322430cb3ca71d8a52aa1e636e88bad8"
VAP_SHA = "f39a78b23a6dccdbedd106e00b48c410b8739f5d"
REPO = "/root/turnbench"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .run_commands(
        f"git clone https://github.com/SesameAILabs/turnbench {REPO} && git -C {REPO} checkout {TURNBENCH_SHA}",
        f"git clone https://github.com/ErikEkstedt/VoiceActivityProjection {REPO}/baselines/vap/VoiceActivityProjection"
        f" && git -C {REPO}/baselines/vap/VoiceActivityProjection checkout {VAP_SHA}",
        f"pip install -e {REPO}",
        "pip install torch==2.7.0 torchaudio==2.7.0 einops tqdm",
        f"pip install -e {REPO}/baselines/vap/VoiceActivityProjection --no-deps",
    )
)

hf_cache = modal.Volume.from_name("turnbench-hf-cache", create_if_missing=True)
app = modal.App("vap-turnbench-dev", image=image)

FLAGS = {"oto": "", "pretrained": "--pretrained", "swbd": "--swbd", "swbd_oto": "--swbd-oto"}


@app.function(
    gpu="A10G",
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": hf_cache},
)
def run_dev(ckpt: str = "oto") -> dict[str, bytes]:
    import os
    import subprocess

    os.environ["HF_HOME"] = "/cache/hf"
    pfx = "" if ckpt == "oto" else f"{ckpt}-"
    subprocess.run(f"bash baselines/vap/run.sh --dev {FLAGS[ckpt]}", shell=True, check=True, cwd=REPO)
    score = subprocess.run(
        ["python", "-m", "turnbench.score", f"baselines/vap/{pfx}predictions-dev.json"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    hf_cache.commit()
    out = {"score.txt": score.encode()}
    for name in ("probs-eot.json", "probs-int.json", "predictions-dev.json"):
        p = Path(REPO, "baselines/vap", pfx + name)
        if p.exists():
            out[pfx + name] = p.read_bytes()
    return out


@app.local_entrypoint()
def main(ckpt: str = "oto"):
    outdir = Path(__file__).parent / "results-modal"
    outdir.mkdir(exist_ok=True)
    for name, data in run_dev.remote(ckpt).items():
        (outdir / (name if name != "score.txt" else f"{ckpt}-score.txt")).write_bytes(data)
    print((outdir / f"{ckpt}-score.txt").read_text())
