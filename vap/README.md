# VAP baseline on TurnBench

Reproduction of the official [TurnBench](https://github.com/SesameAILabs/turnbench) VAP baseline
(Voice Activity Projection, Ekstedt & Skantze), used as this project's starting point.

## Benchmark in brief
- TurnBench (Sesame, arXiv 2608.25218, leaderboard turnbench.sesame.com): per-speaker **end of turn (EOT)**
  and **interruption (INT)** detection on 2-channel dyadic audio. Predictions must be causal.
- A gold event is hit if a prediction lands in `[t-0.25s, min(t+3s, next event)]`. Metrics: recall, false-positive
  rate, latency p10/p50/p90. Ranked by test recall; test FPR must be <= 0.15. The operating point is chosen on dev
  (highest recall at FPR <= 0.10, `turnbench.sweep`).
- Data is gated on Hugging Face (needs `HF_TOKEN` with access approved):
  dev `mundo-ai/turn-benchmark-dev` (38 convs, 7.3 h, labels public),
  test `mundo-ai/turn-benchmark-test` (audio only, scored by Sesame),
  train `otoearth/otoSpeech-full-duplex-turn-104h`.

## The official VAP baseline
`baselines/vap/` in the TurnBench repo: VapGPT (CPC encoder, 50 Hz), 20 s context / 5 s step. The default checkpoint
is VAP fine-tuned on the otoSpeech train set (`viks66/VAP_checkpoints`, `oto`). EOT score = `1 - p_now[spk]`,
INT score = `p_now[spk]`. Thresholds: EOT 0.9161, INT 0.8591.

Published test numbers (`results/official/leaderboard-test.json`):

| model | EOT recall | EOT FPR | EOT p50 | INT recall | INT FPR | INT p50 |
|---|---|---|---|---|---|---|
| vap (oto) | 0.845 | 0.055 | 368 ms | 0.945 | 0.107 | 994 ms |
| espnet_turntaking | 0.826 | 0.078 | 862 ms | 0.573 | 0.080 | 210 ms |
| mimi_endpointer | 0.782 | 0.078 | 645 ms | 0.899 | 0.106 | 1007 ms |
| kyutai_semantic_vad | 0.773 | 0.059 | 1007 ms | 0.898 | 0.081 | 559 ms |

Leaderboard top (EOT, at time of writing): Vox Maru v1 0.960 / 0.072 / 548 ms. VAP's weak spot is INT latency (~1 s).

## Files
- `setup.sh`: clones TurnBench and VAP at pinned commits and installs them (CPU or GPU).
- `modal_dev.py`: runs the oto VAP on the dev set on a Modal A10G and scores it. Results go to `results-modal/`.
- `smoke.py`: speed check with the bundled pretrained checkpoint on 180 s of audio resampled to 16 kHz
  (run from inside the `turnbench/` checkout; 180 s should give 9,000 frames).
- `results/official/`: the TurnBench repo's committed VAP predictions (dev, test) and leaderboard JSON.
  Both prediction files pass `turnbench.check`.

## Running
Locally (CPU is ~3x real time, so dev takes ~2.5 h; a GPU takes minutes):
```
bash setup.sh work && cd work/turnbench
HF_TOKEN=hf_... uv run bash baselines/vap/run.sh --dev        # oto checkpoint
uv run python -m turnbench.score baselines/vap/predictions-dev.json
```
On Modal (from a machine where the Modal CLI can connect):
```
modal secret create huggingface HF_TOKEN=hf_...
modal run modal_dev.py                    # or --ckpt pretrained|swbd|swbd_oto
```

## Scoring on dev without gated data or a GPU
VAP's dev score can be reproduced from public inputs alone:
- dev gold events: https://turnbench.sesame.com/dev-gold.json
- VAP's raw 50 Hz probabilities: HF dataset `freemanjiang/turnbench-baseline-probs` (rev `e3cd4caa`), `vap/probs-{eot,int}.json`

Rescoring those reproduces the official dev result exactly (EOT recall 0.841 at FPR 0.045, INT recall 0.957).
So the gated data and a GPU are only needed to run VAP itself, for example on new audio or a new checkpoint.

## Our dev reproduction (full inference)
Run by Rin on Colab (Tesla T4, 2026-10-06) with `setup.sh` at `cafabe9`: all 38 dev conversations, 10m 31s of
inference (~40x real time). Pins: TurnBench `38a6f87`, VAP `f39a78b`, dev dataset rev `8fa18a2`,
oto checkpoint snapshot `b9aa0ba` (sha256 `8e73c375...db265c`), PyTorch 2.7.0+cu126.

| predictions | task | recall | FP rate | p50 latency | TP / FN / FP / TN |
|---|---|---:|---:|---:|---|
| ours (sweep θ) | EOT | 0.8414 | 0.0452 | 462.5 ms | 1602 / 302 / 48 / 1015 |
| official, rescored | EOT | 0.8409 | 0.0452 | 463.0 ms | 1601 / 303 / 48 / 1015 |
| ours (sweep θ) | INT | 0.9568 | 0.0980 | 911.5 ms | 332 / 15 / 366 / 3367 |
| official, rescored | INT | 0.9568 | 0.0999 | 896.0 ms | 332 / 15 / 373 / 3360 |

The sweep picked EOT θ 0.91614, INT θ 0.86. With the documented thresholds (0.9161 / 0.8591) our probabilities match
the official EOT counts and INT recall, with 374 INT FPs vs 373 (FP rate 0.1002, just over the 0.10 dev budget),
so the outputs are near-identical but not bit-exact. The cause of the residual difference is not known.

## Status
- Dev reproduced both from the public probabilities and by full inference on the gated audio (above).
- Test scores not reproduced (labels are private). `modal_dev.py` has not been run.
