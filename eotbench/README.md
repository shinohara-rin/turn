# LiveKit eot-bench (English) for ssl_turn

Scores our streaming FastConformer floor heads on [LiveKit eot-bench](https://github.com/livekit/eot-bench)
([data](https://huggingface.co/datasets/livekit/eot-bench-data), `en` config, 400 user turns, 84 min,
dataset revision `ca9d98a9`), with LiveKit's own harness, metric and policy sweep, next to the
models LiveKit committed under `output/`.

## Protocol

- eot-bench rows are single user turns (16 kHz mono) with every silence span >= 100 ms. The final
  span is the true end of turn (`eot`), earlier spans are `hold`. Models give `p_eot` on a 0.1 s grid
  over each span, using only audio before that time.
- The harness sweeps threshold x action delay x timeout and reports false-cutoff rate (holds of
  0.2-5 s where the policy fires before the user resumes) vs mean latency on true ends.
- Our models are two-channel. As in eot-bench's VAP adapter, the user goes in channel 0 and the
  agent channel is silent. Score `eot_q` = (p(OPEN) + p(HELD_agent)) x p(user SILENT), our TurnBench
  default (score.py).
- `modal_eot.py` encodes each turn once (causal FastConformer, 80 ms frames) and runs the head over
  the whole turn. `causality_check` confirms this equals scoring each audio prefix separately (100
  random cuts, max |diff| 0.006, bf16 noise). `to_harness.py` reads frame floor(t / 0.08) - 1 at
  grid time t and reproduces the harness grid exactly (11,191 rows, same as the committed VAP run).

## Results (en, eot_q, r019 = `r019_asr_bgaug` checkpoints)

| Model | Cutoff @ 300 ms | Cutoff @ 600 ms | Latency @ 5% cutoff | Latency @ 10% cutoff | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiveKit Turn Detector v1 | 9.9% | 4.5% | 543 ms | 295 ms | |
| JoinIn AI Baton | 12.3% | 4.8% | 577 ms | 350 ms | |
| Soniox | – | 5.5% | 647 ms | 512 ms | |
| **r019 bgaug_s2** | 21.0% | 8.4% | 805 ms | 539 ms | 0.975 |
| **r019 bgaug_s1** | 22.6% | 9.1% | 815 ms | 570 ms | 0.968 |
| **r019 fine1_bal1_s1** (clean) | 18.9% | 8.2% | 826 ms | 516 ms | 0.968 |
| **r019 fine1_bal1_s2** (clean) | 20.0% | 8.7% | 838 ms | 524 ms | 0.972 |
| ultraVAD | 27.7% | 11.9% | 899 ms | 663 ms | |
| LiveKit Turn Detector v1-mini | 27.8% | 12.1% | 1070 ms | 698 ms | 0.890 |
| SmartTurn v3.2 | 35.2% | 14.8% | 1051 ms | 739 ms | 0.845 |
| VAP (silent agent) | 47.0% | 14.6% | 1131 ms | 749 ms | 0.934 |
| Deepgram Flux | 12.9% | 9.9% | 1151 ms | 548 ms | |
| VAD baseline | 55.6% | 21.7% | 1600 ms | 1000 ms | |

Published rows are LiveKit's committed artifacts; recomputing VAP's metrics with the harness matches
them exactly. Other variants (`eot`, `eot_f04_q`) land in the same 776-846 ms band at 5% cutoff.

- Zero-shot (no eot-bench training, English telephone-style oto data only) the heads place 4th of 13
  English systems, behind LiveKit v1, Baton and Soniox, and ahead of every other open model
  (SmartTurn, ultraVAD, LiveKit v1-mini, VAP) and of Deepgram Flux, AssemblyAI, Gradium, Cartesia and
  GPT Realtime at the 5% budget.
- bg_aug costs nothing here (seed spread ~30 ms is larger than the clean vs bg_aug gap).
- The gap to LiveKit v1 is at the low-latency end: at 300 ms we cut off ~20% of pauses vs 9.9%.
  Our heads see audio only; v1 and Baton also use the conversation text, and eot-bench gives no agent
  audio, so our cross-channel features are idle.
- The turn-1-mini-style rules policy is VAD + timing only. On eot-bench the silence spans are given,
  so any rule that only looks at silence timing is the harness's VAD baseline (1600 / 1000 ms). Its
  TurnBench gains came from TurnBench's scorer (fires after a cancelled candidate, confirmation
  events), which eot-bench does not have: the first fire in a pause decides.

## Reproduce

```bash
pip install 'modal[api-proxy-support]'
modal run eotbench/modal_eot.py --out tracks.npz            # ~80 s on an L4, needs ssl-turn-work/runs/r019_asr_bgaug
git clone https://github.com/livekit/eot-bench && pip install -e eot-bench
cp -r eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/en en
python eotbench/to_harness.py tracks.npz en --variants eot_q
for d in en/ssl_turn__*; do eot-harness compute-metrics --predictions $d/predictions.parquet --output-dir $d/metrics; done
eot-harness compare-models en
```

(The harness pins numpy<2; pyarrow 18.1 works with it.)
