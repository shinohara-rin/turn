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
| r011 `@r0.5+rc1.0` | 3 seeds of mid_reg_no7 (131 conversations; same batches, different init/dropout) | 0.941 / 0.930 / 0.935 (**0.935 ± 0.006**), p50 ~230 ms | 0.977–0.983, p50 ~520–550 ms |

### Backbone comparison (same 23 conversations, fixed 500 steps, no early stopping)

| Backbone | EOT `eot_q@r0.5+rc1.0` | INT `int_spk@r0.5` |
|---|---|---|
| Cat (causal codec encoder, 80 ms) | small 0.884 / 0.099 / 468 ms; mid 0.859 / 0.100 / 482 ms | small 0.971 / 0.096 / 596 ms; mid 0.960 / 0.098 / 554 ms |
| MOSS-Transcribe-Diarize (trailing 30 s windows every 160 ms) | small 0.895 / 0.099 / **301 ms**; mid 0.869 / 0.099 / 304 ms | small 0.974 / 0.099 / **421 ms**; mid 0.977 / 0.100 / 412 ms |
| Cat + MTD concatenated | small 0.882 / 0.100 / 341 ms; mid 0.847 / 0.094 / 440 ms | small 0.971 / 0.096 / 392 ms; mid 0.971 / 0.091 / 423 ms |

MTD matches Cat's recall and commits about 150–170 ms earlier on both tasks, despite a
2× coarser update. Fusion didn't help at this data size. Exactly causal MTD features cost
about 37 channel-s/s on an H100, at 100% utilization (one 30 s pass per 160 ms step).

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

## Residual analysis at the official dev operating point

`score.py::residuals` uses TurnBench dev's three-annotator segments and transcripts and
compares each factor against its base rate. Its window matching approximates the
scorer: 109 misses and 68 FPs, where the official counts are 116 and 72.

**EOT misses (about 5.7% of 1904):**
- *Lapses:* if the other speaker's next turn starts more than 1 s later, misses are 9.3%
  (about 4% otherwise). With no following turn ("Okay.", "Bye-bye", call endings), misses
  are 35% (15/43). Until someone speaks, these look like holds, so they are largely
  irreducible causally.
- *Early fires:* 50/109 misses had a fire 0.25–3 s before the annotated end, often at a
  pause before a final phrase or a trailing laugh.
- *Not factors:* overlapping hand-offs (4.2–4.3%), annotator agreement (nearly all 3/3),
  and conversation type (4.8–8.6%).

**EOT false fires (6.4% of 1063 holds):**

| Hold length | <0.3 s | 0.3–0.6 s | 0.6–1.2 s | 1.2–2.5 s | >2.5 s |
|---|---|---|---|---|---|
| FP rate | 0.8% | 4.5% | 7.9% | 24.5% | 62.5% |

- Long holds are the mirror image of lapses, and many follow syntactically complete
  sentences.
- A listener backchannel inside the hold raises FPs 3.5× (13.5% vs 3.9%). The model
  reads it as floor take-up, though backchannels never move the floor in gold, so this
  is a fixable model error.
- Holds after an interruption-won turn run 13–17%.
- A trailing function word ("and", "so", "um") does *not* lower FPs (5.7% vs 6.5%): the
  Cat features carry no lexical-incompleteness cue.

**INT false fires (8.4% of 3733 backchannel/noise spans):**
- Reaction backchannels 35%, acknowledgements 20%, continuers 6.5%, against noise 1.1%,
  bleed 0.6% and non-linguistic 2.6%. So the errors are lexical backchannels that sound
  like turns.
- Backchannels into silence (the main speaker paused) fire 36%.
- Spans whose majority fine label is "Normal Turn" or "Bounded Response" fire about 50%;
  that is label ambiguity.

**INT misses (9):** mostly laughter-initial interruptions.

**Summary:**
- *Inherent to causal detection plus TurnBench's conventions:* about 45 EOT misses
  (lapses, no reply) and about 30 FPs (holds over 1.2 s).
- *Targetable:* backchannel-during-hold confusion, turn-like reaction backchannels,
  laughter onsets, and early fires before trailing laughs. These point to semantic
  features and to emphasizing those cases in training, not to more data of the same kind.

## Fine annotator labels (r012)

`labels.FINE` gives each speaker frame one of 17 annotator labels (plus SILENT), so
backchannel and interruption subtypes, Strong Floor Hold, Bounded Response, Awkward
Silence and so on are supervised directly. The fine head is per-speaker (per-slot in
mono), with optional inverse-frequency balancing (weight ∝ freq^-α, clipped at 20×).

Setup: 4 arms × 2 seeds on 131 conversations. With FP capped at 0.10, recall is saturated
(INT ~0.98, EOT ~0.93), so arms are compared by **FP at fixed recall** (mean of 2 seeds).

| INT FP (p50) | recall ≥ 0.95 | recall ≥ 0.97 |
|---|---|---|
| no fine head, `int_spk` | 0.032 (847 ms) | 0.060 (668 ms) |
| fine, α=1, `int_nobc` | **0.016** (878 ms) | **0.046** (628 ms) |
| fine, α=0.5, `int_nobc` | 0.023 (862 ms) | 0.048 (649 ms) |
| fine head, scored with plain `int_spk` | 0.033–0.041 | 0.082–0.084 |

`int_nobc` = `int_spk` × (1 − the speaker's own predicted backchannel + noise mass).

- **The gain comes through scoring:** the fine head alone, scored the old way, does not
  reduce INT false fires.
- **It lands where the residuals pointed** (control vs fine α=1 at matched recall 0.971 /
  0.974): reaction backchannels 0.294 → 0.188, acknowledgements 0.157 → 0.115,
  continuers 0.043 → 0.034, backchannels into silence 0.279 → 0.135. INT false fires
  total 240 → 177 (−26%). Non-speech noise rises slightly (4 → 11 fires).
- **Earlier INT detection:** `int_ft` (the floor-taking subtype posterior alone) fires
  much earlier (p50 ~260–330 ms vs ~600 ms) at higher FP. That is a latency/precision
  knob.
- **EOT doesn't improve:** FP at recall 0.92 is 0.053–0.061 in every arm.
  Suppressing EOT while the listener backchannels (`eot_nobc`) is worse, because
  backchannels also follow real yields. The listener-backchannel EOT confusion needs a
  training-side fix, not gating.

## LLM text oracle (upper bound for fusing an LM)

Question: how much could conversational understanding from text add to the audio model?
Setup (`pipeline/llm_oracle.py`, `score.oracle_fusion`):
- **Text:** annotator transcripts stand in for a perfect streaming ASR. Words are revealed
  uniformly over each segment, with 0.3 s latency. Only text and timing are used, never labels.
- **Queries:** a hosted 30B LLM (thinking off) gives P(yes) from first-token logprobs.
  - EOT: asked at each segment end + 0.3 s.
  - INT: asked at 1/2/4/all words of segments that start during the other speaker's speech.
  - 25.6k queries over TB dev + oto dev.
- **Fusion:** geometric `audio^(1-w) · text^w`, with text = 0.5 wherever no query is live.
  The audio model is r012 fine α=1, 2 seeds.
- **Control:** the same query times with answers shuffled across queries of the same kind.

TB dev, FP at fixed recall:

| | w | real text | shuffled text |
|---|---|---|---|
| INT @R0.97, s1 | 0 | 0.049 | 0.049 |
| | 0.5 | 0.025 | 0.019 |
| | 0.75 | 0.013 | 0.016 |
| INT @R0.97, s2 | 0 | 0.044 | 0.044 |
| | 0.5 | 0.021 | 0.020 |
| | 0.75 | 0.015 | 0.015 |
| EOT @R0.92 | 0.5 | ~0.21 (from 0.054) | same |

- **The naive INT gain (3× fewer FPs) is a timing leak.** Query times sit on annotated
  segment boundaries, so "text exists here" marks real speech onsets. Shuffled answers get
  almost all of the gain. The content-specific increment is about zero.
- **EOT fusion trades precision for latency, and content adds nothing over shuffled text.**
- **Takeaway:** with naive fusion, a text LM's content does not reduce the residual
  false fires. A fair test needs text queries on the model's own (VAD/onset) timeline,
  reported against the shuffled baseline, and probably a learned fusion. Transcripts and LLM
  outputs stay off git (scratch + Modal volume only).

### Follow-up: LLM as a verifier on the audio model's fires (stopped)

`pipeline/llm_verifier.py` puts the queries on the audio model's own fires, so no gold timing
leaks. The transcript mimics streaming ASR (0.3 s lag). The question is plain ("is A done?" /
"is A taking the floor?"), with a sentence or two of visible reasoning, and the answer is a
yes/no line. Model thinking was turned off because it ran past 2k tokens.

Sanity check on the first ~3.5k EOT answers (10 TB dev conversations, loose threshold):

| keep fire if | real: recall / FP | shuffled: recall / FP |
|---|---|---|
| (audio only) | 0.948 / 0.318 | |
| answer = yes | 0.704 / 0.188 | 0.771 / 0.211 |
| P(yes) >= 0.8 | 0.592 / 0.152 | 0.693 / 0.186 |

- **Worse than random gating.** The LLM says "done" 61% of the time at true ends and 69% at
  mid-turn pauses (last words visible).
- **Lag:** 28% of true-end fires come before the final words clear the ASR lag.
- **Prompt-shape problems:** the listener's later backchannel follows the speaker's
  unfinished line, and long annotator segments look complete.

Stopped at ~7k of 32.8k prompts. Bolting an LM onto the scores this way looks like a dead end.

## Human ceiling: annotator agreement under TurnBench scoring

`score.human_ceiling`. Each TB dev annotator's own labels are treated as a system: EOT fires
at their turn ends, INT fires at their floor-taking interruption onsets. These fires have
zero latency and full hindsight. They are scored with TurnBench's scorer against gold rebuilt
from the other two annotators (2-of-2). Rebuilding the gold with all three annotators
reproduces the official gold exactly. The audio model (r012 fine α=1 s1, official operating
point) is scored against the same leave-one-out golds.

| vs. the other two annotators | EOT recall / FP | INT recall / FP |
|---|---|---|
| annotator a | 0.894 / 0.119 | 0.647 / 0.000 |
| annotator b | 0.877 / 0.093 | 0.686 / 0.000 |
| annotator c | 0.822 / 0.092 | 0.527 / 0.000 |
| model (mean of 3 golds) | 0.937 / 0.080 | 0.979 / 0.055 |

**EOT is at the label-agreement ceiling.**
- The model agrees with any two annotators better than the third annotator does.
- Errors by annotator agreement (official gold):
  - Recall is 0.955 on unanimous ends and 0.911 on 2-of-3 ends; the 2-of-3 ends hold
    about half the misses.
  - In mid-turn pauses where one annotator marked an end, FP is 0.111, vs 0.059 where none
    did. Those spans hold only ~8 of the 71 false fires.
- So most EOT false fires sit in pauses that every annotator calls a hold. The model is
  wrong there, not the labels.

**INT false fires are not label noise.**
- No annotator ever marks a backchannel or non-content span as an interruption (FP ≈ 0).
- Annotators mostly disagree on positives, i.e. which category an overlap belongs to
  (recall 0.53–0.69).
- The model's INT false fires (backchannels) are a causal problem: at ~400 ms it cannot
  yet hear what a human hears with hindsight. Headroom there is real, but it is a
  latency/evidence trade-off, not a labeling one.

## Encoder fine-tuning: LoRA on Cat top layers 16-31 (r013)

`cat_top.py` re-runs Cat's top-stage layers 16-31 and its output projection from cached
tap 15. The setup:
- **Exactness:** RoPE is relative, and each crop gets 125 frames (10 s) of context, so
  outputs match the full encode. At init the re-run matches the cache (cosine 1.000 mean,
  0.98 min, bf16).
- **Adaptation:** LoRA rank 16 on every linear map (8.4M trainable params). Frozen weights
  are stored in bf16. The head is fine1_bal1, lr 1e-4 for LoRA.
- **Run:** 131 conversations, 1000 steps × 64 crops, 2 seeds trained in lockstep.
- **Cost:** H100 at 1.38 s/step with 99% utilization, 71 GB peak (tap 15 + final only in
  VRAM, 12.7 GB). About $1.9 for training and inference.

| TB dev | EOT (`eot_q@r0.5+rc1.0`) FP@R0.92 | EOT FP@R0.94 | INT (`int_nobc@r0.5`) FP@R0.95 | INT FP@R0.97 |
|---|---|---|---|---|
| head only (r012, s1 / s2) | 0.054 / 0.061 | 0.127 / 0.132 | 0.014 / 0.018 | 0.048 / 0.044 |
| LoRA top (r013, s1 / s2) | 0.073 / 0.067 | 0.160 / 0.097 | 0.016 / 0.016 | 0.033 / 0.045 |

- **Dev loss:** floor+future improves slightly, 0.664 → 0.658. Both seeds were still
  improving at the last step.
- **TB dev:** no consistent change; every difference is within seed noise.
- **Next:** r014 trains 2000 steps with LoRA lr 2e-4.

## Cost

About $11–12 of the $20 allocation for all of the above (Modal billing for `ssl-turn-*`
apps: H100 ~$3.6, A100-80GB ~$3, A100-40GB ~$0.8, L4 ~$0.75, CPU + memory ~$3.2).
The largest avoidable costs, all fixed:
- **TurnBench dev decode:** each of 8 CPU containers read the whole 4.2 GB parquet.
- **Trainer RAM reservation:** 64 GB reserved while features lived in VRAM.
- **MTD per-item volume commits:** about 30 s of an idle H100 per item.

Training runs sit at 94–98% GPU utilization by training all configs in lockstep on
VRAM-resident features. Feature loading went from 255 s to 146 s for 1.2× the data with
parallel, preallocated reads.
