#!/usr/bin/env bash
# Reproduce the TurnBench + VAP environment (CPU or GPU).
# Needs: network access to github.com, pypi.org, huggingface.co,
#        and HF_TOKEN with access to the gated mundo-ai/turn-benchmark-* datasets.
set -euo pipefail
DIR="${1:-$PWD/work}"; mkdir -p "$DIR"; cd "$DIR"
git clone https://github.com/SesameAILabs/turnbench && git -C turnbench checkout 38a6f874322430cb3ca71d8a52aa1e636e88bad8
git clone https://github.com/ErikEkstedt/VoiceActivityProjection turnbench/baselines/vap/VoiceActivityProjection
git -C turnbench/baselines/vap/VoiceActivityProjection checkout f39a78b23a6dccdbedd106e00b48c410b8739f5d
cd turnbench
uv sync
uv pip install torch==2.7.0 torchaudio==2.7.0 soundfile einops tqdm
uv pip install -e baselines/vap/VoiceActivityProjection --no-deps
# Then (with HF_TOKEN set):
#   uv run bash baselines/vap/run.sh --dev               # oto-finetuned ckpt (official baseline)
#   uv run bash baselines/vap/run.sh --dev --pretrained  # original Switchboard VAP ckpt
#   uv run python -m turnbench.score baselines/vap/predictions-dev.json
