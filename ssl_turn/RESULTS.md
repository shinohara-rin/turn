# Attempt 2 experiment log

All runs are on Modal (`ssl-turn-*` apps; volume `ssl-turn-work`). Features come from the
frozen causal Cat encoder (TF32, taps of top-stage layers 7/15/23/31 + the final output,
80 ms grid). Heads are floor-ownership models (`model.py`, `labels.py`), trained on the
frozen otoSpeech actor split. Training uses 131 train conversations unless noted;
early stopping is on otoSpeech dev loss.

## Scoring

The pinned evaluator's `commit_events` (rising edge) and `score_task` are used unchanged.
- **Sweep:** 256 score quantiles plus a 0.01 grid.
- **Operating point:** highest recall with FP ≤ 0.10, as the TurnBench baselines choose it.
- **Data:** TurnBench numbers are **dev**, swept on dev (development evidence, not test).
  otoSpeech dev, with single-annotator gold built by the same gold builder, is a
  secondary selection set. Its thresholds do not transfer to TurnBench dev (different
  annotation consensus), so TurnBench operating points are chosen on TurnBench dev.

The commit refractory is the official sweep's 2 s unless marked `@r0.5` (0.5 s). A
submission may use any causal commit policy; the refractory is part of the policy.

Score definitions (from floor posteriors; `score.py::score_variants`):

| Name | Definition |
|---|---|
| `eot` | p(`OPEN`) + p(`HELD_other`) now |
| `eot_q` | `eot` × p(c `SILENT`) from the act head |
| `int_f04` | p(`HELD_c`) 0.4 s ahead |
| `int_spk` | `int_f04` × (1 − p(c `SILENT`)) |

Reference points on TurnBench **dev**:
- VAP (official baseline): EOT 0.841 / FP 0.045 / p50 462 ms; INT 0.957 / 0.098 / 912 ms.

Ooma's **test** numbers (EOT 0.951 / 0.082 / 388 ms, INT 0.968 / 0.076 / 567 ms) are not
directly comparable to dev.

## Runs

| Run | Change | Best TurnBench dev EOT (R / FP / p50) | Best TurnBench dev INT |
|---|---|---|---|
| r001 | 32 train conversations, 6 configs, no early stopping (overfit) | 0.709 / 0.100 / 638 ms | 0.732 / 0.100 / 1218 ms |
| r002 | 131 conversations, regularization grid, early stopping | 0.803 / 0.020 / 521 ms (`eot`) | 0.963 / 0.087 / 694 ms |
| r003 | top configs, posteriors saved; transition scores (worse) | 0.799 / 0.022 / 443 ms (`eot`) | 0.957 / 0.097 / 664 ms (`int_f04`) |
| r003q | r003 checkpoints re-inferred with act posteriors | 0.827 / 0.036 / 475 ms (`eot_q`) | 0.977 / 0.083 / 551 ms (`int_spk`) |
| r003q `@r0.5` | 0.5 s commit refractory | **0.892 / 0.099 / 250 ms** (small_reg `eot_q`) | **0.980 / 0.099 / 498 ms** (small_reg `int_spk`) |
| r004 `@r0.5` | tap ablation, second seed, ensembles (131 conversations) | 0.904 / 0.065 / 310 ms (mid_reg_no7) | 0.986 / 0.098 / 551 ms (small_reg_t31) |
| r005 `@r0.5` | **231 conversations** (+100 gate-free extra) | 0.892 / 0.092 / 220 ms | 0.986 / 0.097 / 495 ms (ensemble) |
| r006 `@r0.5` | **32 conversations**, same pipeline | 0.897 / 0.100 / 356 ms | 0.983 / 0.096 / 577 ms |
| r007 `@r0.5` | objective ablation: no VAP, 3× projection, 2× act | 0.882–0.890 (all within noise) | 0.977–0.980 |
| r008 `@r0.5` | temporal label smoothing (3/5/9 frames) | 0.878–0.897 (within noise) | 0.977–0.983 |
| r004 `@r0.5+rc1.0` | **re-commit** once the score stays above θ for 1 s | **0.942 / 0.091 / 246 ms** | — |

### Official dev operating point (`runs/r004/predictions-dev-rc.json`)

Validated by `turnbench.check` and scored by `turnbench.score` (pinned 38a6f87); the
official scorer reproduces the sweep exactly.

| Task | Model / score / policy | Recall | FP | p10 / p50 / p90 |
|---|---|---|---|---|
| EOT | r004 mid_reg_no7, `eot_q`, θ 0.81, 0.5 s refractory + 1 s re-commit | **0.939** | 0.068 | −53 / 296 / 1031 ms |
| INT | r004 small_reg_no7, `int_spk`, θ 0.415, 0.5 s refractory | **0.974** | 0.085 | 150 / 585 / 1572 ms |

The thresholds were picked on TurnBench dev with a margin under the 0.10 FP budget
(recall is flat there: EOT 0.942 at FP 0.091 vs 0.939 at 0.068). This is dev evidence
only; no test audio has been touched.

## What the probes showed

- **More data helps most:** 32 → 131 conversations moved EOT 0.71 → 0.80 and INT 0.73 → 0.96.
  Every config peaks at 500–1000 steps (about 10 epochs of crops), so data, not model
  size, is the limit. The dim 128/2-layer and dim 256/4-layer heads are within noise.
- **EOT recall was capped by the rising-edge commit rule, not by FPs.** Recall plateaued
  near 0.80 with FP at 0.02; lowering θ lost recall. Miss anatomy at the operating θ
  (r003 `eot`): 154 pre-fired (the score already rose more than 0.25 s before the turn
  end, while the speaker was finishing), 91 refractory-suppressed, 115 never crossed.
  - **Fix 1:** gating by the act head's p(silent) halved pre-fired misses (154 → 75).
  - **Fix 2:** a 0.5 s refractory recovered most refractory-suppressed misses.
  - Latency fell to ~250 ms in the process.
- **"Transition" scores** (released now × held recently) cut latency but lost recall;
  dropped.
- **otoSpeech-dev thresholds don't transfer:** they over-fire on TurnBench dev for EOT
  and under-fire for INT, so the operating point must come from TurnBench dev.
- **Configs are saturated:** head size, window, regularization and tap choice all land
  within ±0.006 EOT recall, about the gap between two seeds of one config. Tap 7 adds
  nothing; tap 31 + final alone is within noise. Posterior ensembles add nothing.
- **Data volume stopped mattering once the commit policy was fixed:** 32, 131 and 231
  conversations all reach EOT about 0.89–0.90 and INT about 0.98 on TurnBench dev; more
  data only lowers latency (about 350 → 250 ms). The remaining EOT gap is not otoSpeech
  quantity.
- **Remaining errors:** at the best EOT point (r004 mid_reg_no7, `eot_q@r0.5`), 167 misses
  are 81 pre-fired, 76 never crossed (peak 0.58–0.80) and 10 refractory. False fires land
  in long holds (median 1.03 s vs 0.40 s for clean pauses), which is where semantic
  completeness should help. INT false fires are long backchannel/noise spans (median
  0.98 s).
- **Noise floor:** the same config (mid_reg_no7) scored EOT 0.904 in r004 and 0.884 in
  r007/r008. Differences under about 0.02 are not significant without repeated seeds.
- **Re-commit fixes the pre-fired misses:** when the score crosses before the turn ends
  (the model is right but early) and stays high, firing again after 1 s of continuous
  score above θ lands inside the gold window. EOT recall 0.904 → 0.942 at similar FP and
  lower p50. Pause FPs grow by at most one per long hold. This mirrors the "delayed
  re-commit while the channel stays silent" in Ooma's description.
- **Encoder precision:** pure bf16, and also bf16 autocast, drift on a few frames (cosine
  down to 0.88 against fp32). TF32 is within 0.9997, so features are cached with TF32.

## Cost

Running total for this attempt's apps is tracked with `modal billing report`.
