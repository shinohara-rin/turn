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
- turn-1-mini-style policy (rules alone): VAD + timing only. On eot-bench the silence spans are
  given, so any rule that only looks at silence timing is the harness's VAD baseline (1600 / 1000 ms).

## turn-1-mini policy + r019 (`t1m_policy.py`)

The round-2 hybrid from the "Study turn-1-mini" thread commits a pause at min(VAD deadline, first
time eot_q >= th at or after t + mw_model). That is the harness's own threshold / action delay /
timeout policy except for one detail: the harness fires at max(delay, first crossing), so a score
that crossed early and fell back still fires at the delay, while turn-1-mini reads the score only
once the delay has passed. `t1m_policy.py` sweeps both rules on the harness grid (its `harness`
rule reproduces `eot-harness compute-metrics` exactly for our runs and VAP).

| eot_q head | rule | Latency @ 5% | Latency @ 10% | Cutoff @ 300 ms | Cutoff @ 600 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| bgaug_s1 | harness / t1m | 815 / 813 ms | 570 / 526 ms | 22.6 / 22.1% | 9.1 / 8.4% |
| bgaug_s2 | harness / t1m | 805 / 804 ms | 539 / 524 ms | 21.0 / 20.1% | 8.4 / 8.2% |
| fine1_bal1_s1 | harness / t1m | 826 / 818 ms | 516 / 507 ms | 18.9 / 18.3% | 8.2 / 7.8% |
| fine1_bal1_s2 | harness / t1m | 838 / 838 ms | 523 / 501 ms | 20.0 / 19.0% | 8.7 / 7.8% |

The turn-1-mini rule helps a little on every head (0-8 ms at 5%, 9-44 ms at 10%, about 0.5-1 point
fewer cutoffs) but does not change the ranking (Soniox 647 / 512 ms is next above). The rest of the
policy has nothing to act on here: the 2.5 s confirmation event and the resumption cancel only matter
after a first fire, and on eot-bench the first fire in a pause already decides a cutoff; the
other-speaker trigger needs agent audio; and the pauses are the dataset's spans, not our VAD. The
TurnBench deadlines (< 1 s) are also below the harness's timeout grid (1-3.5 s); the VAD baseline at
1.0 s already cuts off 21.7% of pauses, so shorter deadlines cannot help at a 5-10% budget.

## Reproduce

```bash
pip install 'modal[api-proxy-support]'
modal run eotbench/modal_eot.py --out tracks.npz            # ~80 s on an L4, needs ssl-turn-work/runs/r019_asr_bgaug
git clone https://github.com/livekit/eot-bench && pip install -e eot-bench
cp -r eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/en en
python eotbench/to_harness.py tracks.npz en --variants eot_q
for d in en/ssl_turn__*; do eot-harness compute-metrics --predictions $d/predictions.parquet --output-dir $d/metrics; done
eot-harness compare-models en
python eotbench/t1m_policy.py en/ssl_turn__*_eot_q/predictions.parquet
```

(The harness pins numpy<2; pyarrow 18.1 works with it.)
