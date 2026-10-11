# Semantics, silence-driven EOT, and backchannels (2026-10-11)

Rin's playground report: the r019 EOT score tracks time since the last voice activity, so EOT is
always late, hesitant speakers get cut off mid-sentence, and floor taking does not handle
backchannels well, possibly from overfitting to otoSpeech speakers. This folder tests those claims.

## 1. EOT climbs with silence (LiveKit eot-bench en, saved r019 predictions)

The score on true ends and on mid-turn holds at a fixed silence length (400 turns, 850 holds). Holds
thin out with silence, as in the data. LiveKit rows come from eot-bench's committed predictions.

| silence | r019 bgaug_s2 hold / end | AUC | LiveKit v1 hold / end | AUC | v1-mini AUC | VAP AUC |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 0.1 s | 0.05 / 0.25 | 0.848 | 0.19 / 0.92 | 0.961 | 0.880 | 0.746 |
| 0.3 s | 0.09 / 0.44 | 0.896 | 0.14 / 0.90 | 0.971 | 0.889 | 0.801 |
| 0.5 s | 0.12 / 0.52 | 0.906 | 0.17 / 0.89 | 0.966 | 0.804 | 0.831 |
| 1.0 s | 0.25 / 0.67 | 0.890 | 0.21 / 0.89 | 0.953 | 0.530 | 0.843 |
| 1.5 s | 0.41 / 0.76 | 0.895 | 0.21 / 0.87 | 0.949 | 0.625 | 0.803 |

r019 separates ends from holds (AUC 0.85-0.92 at a fixed silence), but both curves climb with
silence, while LiveKit v1 (audio-only) decides within 100 ms and stays flat. The climb is the delay,
and long holds drifting to ~0.4 are the mid-sentence cutoffs.

## 2. Where the end-of-turn cue is: pause-onset probes (`probe_eotbench.py`, `probe_fit.py`)

End vs hold at a fixed silence length, 5-fold CV grouped by turn, standardized + PCA 64 + L2
logistic regression. Probes are fit on eot-bench itself, so they are ceilings for each input, not
zero-shot numbers. `text model` is LiveKit's text turn detector (v0.4.1-intl, ONNX) on the
dataset's true words before the pause, with the agent context (`text_probe.py` in the scratchpad
recipe below).

| input | 0.1 s | 0.3 s | 0.6 s |
| --- | ---: | ---: | ---: |
| r019 eot_q, zero-shot | 0.848 | 0.896 | 0.907 |
| r019 head hidden state (192-d) | 0.898 | 0.930 | 0.919 |
| FastConformer frame (mid + output, 1024-d) | 0.909 | 0.936 | 0.933 |
| FastConformer, mean of last 0.4 s of speech | 0.875 | 0.860 | 0.846 |
| FastConformer frame + last speech | 0.915 | 0.931 | 0.933 |
| RNNT prediction net state (640-d) | 0.618 | 0.649 | 0.695 |
| RNNT joint hidden (640-d) | 0.725 | 0.726 | 0.786 |
| FastConformer + RNNT pred + joint | 0.906 | 0.931 | 0.929 |
| text model, true words | 0.757 | 0.763 | 0.778 |
| r019 eot_q + text | 0.873 | 0.918 | 0.938 |
| FastConformer + RNNT + text | 0.907 | 0.926 | 0.927 |
| LiveKit v1 (reference, zero-shot) | 0.961 | 0.971 | ~0.966 |

- The encoder already carries more than the head delivers at pause onset (0.91 vs 0.85).
- Text adds nothing once the encoder is used. A purpose-built text EOU model with perfect words
  only reaches 0.76: users pause at grammatical boundaries ("I am glad for your help today ...
  and this is all I need"), so words alone cannot separate those holds from ends. It helps r019's
  raw score only because the head under-uses its encoder. Same verdict as the TurnBench fusion test
  (`textfuse/`): a DistilBERT-style text branch on the decoder output is not the lever.
- The RNNT decoder state is weak: the prediction net is a small token LM; the content is in the
  encoder.

## 3. Backchannels (`bc_check.py`, r019 TB dev / oto dev probabilities)

Per annotator-a backchannel while the other speaker holds a turn; thresholds are the playground's
(INT 0.18, EOT 0.64). `int_fire`: the backchannel's own INT score crosses; `eot_fire`: the main
speaker's EOT crosses although they keep talking.

| model | split | backchannels | INT fires | main-speaker EOT fires |
| --- | --- | ---: | ---: | ---: |
| bgaug_s2 | oto dev (held-out speakers) | 803 | 25.9% | 13.3% |
| bgaug_s2 | TB dev (other corpus) | 1487 | 10.6% | 12.3% |
| bgaug_s1 | oto dev | 803 | 26.3% | 13.2% |
| bgaug_s1 | TB dev | 1487 | 13.5% | 10.7% |

INT fire rate by backchannel length in words (bgaug_s2): TB dev 1 word 4.4%, 2 words 15.6%,
3 words 17.8%, 4+ words 48.3%; oto dev 17% / 40% / 50% / 84%.

- Not an otoSpeech-speaker overfit: new TB speakers fire less than held-out oto speakers.
- The failure is lexical: one-word "mm-hmm / yeah" mostly passes, while multi-word reactions ("oh my
  god, absolutely", "yes, exactly") are treated as floor takes. TB annotators label them
  backchannels by content, which the head does not read reliably at ~400 ms.
- About 1 in 8 backchannels also lifts the speaker's EOT score over threshold (a listener's
  vocalisation reads as a take-over).

## 4. Pause-warp augmentation (r020, `pipeline/pausewarp.py`)

Rin's proposal: adversarial augmentation so silence length stops predicting the label. Warped copies
of the 131 training conversations stretch mid-turn holds (p 0.5, +0.3-2.0 s of the gap's own room
tone, 5 ms crossfades) and shorten yield gaps (p 0.5, to 0.16 s up to their length); half of the
training crops come from the warped copies (`warp_aug: 0.5`, `configs/r020_warp.json`). In the
original data, holds are shorter than yield gaps (median 0.44 vs 0.56 s on conversation 270).

Cost: warp + encode 6 x L4 ~$0.39, training (H100, 1000 steps, 2 seeds) ~$0.17, eot-bench ~$0.13.

| | eot-bench AUC | latency @2% / 5% / 10% cutoff | TB dev EOT (eot_q, r0.5 + rc1.0) | TB dev INT (int_nobc) | oto dev floor+future |
| --- | ---: | --- | --- | --- | ---: |
| r019 bgaug_s1 | 0.968 | 1608 / 815 / 570 ms | 0.934 / FP 0.100 / 284 ms | 0.980 / 0.088 / 513 ms | 0.671 |
| r019 bgaug_s2 | 0.975 | 1660 / 805 / 539 ms | 0.936 / 0.099 / 286 ms | 0.986 / 0.096 / 479 ms | 0.679 |
| r020 warp_s1 | 0.963 | 1556 / 838 / 548 ms | 0.935 / 0.097 / 303 ms | 0.983 / 0.099 / 468 ms | 0.670 |
| r020 warp_s2 | 0.966 | 1454 / 840 / 547 ms | 0.932 / 0.100 / 276 ms | 0.977 / 0.085 / 517 ms | 0.671 |

Hold / end score and AUC at a fixed silence on eot-bench:

| | 0.1 s | 0.3 s | 0.5 s | 1.0 s | 1.5 s |
| --- | --- | --- | --- | --- | --- |
| r019 bgaug_s2 | 0.05 / 0.25, 0.848 | 0.09 / 0.44, 0.896 | 0.12 / 0.52, 0.906 | 0.25 / 0.67, 0.890 | 0.41 / 0.76, 0.895 |
| r020 warp_s2 | 0.08 / 0.32, 0.851 | 0.13 / 0.52, 0.906 | 0.17 / 0.60, 0.913 | 0.28 / 0.67, 0.892 | 0.38 / 0.71, 0.889 |

On oto dev (in-domain, held-out speakers) holds still climb too: r019 0.09 -> 0.34 and r020
0.10 -> 0.30 from 0.08 to 1.5 s of silence, hold-vs-yield AUC 0.73-0.80 for both.

**Verdict: neutral.** Scores rise earlier on true ends, but holds rise with them. TB dev and oto
dev are unchanged, eot-bench is unchanged at the 5% and 10% budgets, and only the 2% budget
improves (the long-timeout regime, 1454-1556 vs 1608-1660 ms). The hold ramp is barely flatter.
Likely reasons, untested: the inserted room tone repeats the gap's own frames, which may be a
recognizable "edited pause" cue (a new shortcut), and every silent frame weighs the same in the
loss, so long easy silences dominate over the 100-300 ms decision.

## Takeaways

- Rin's diagnosis holds: EOT evidence accumulates with silence instead of being decided at pause
  onset. The encoder has more onset information than the head uses (0.91 vs 0.85 AUC at 100 ms),
  so the head's objective and data are the place to work, not a text branch or the RNNT decoder.
- Backchannel INT fires are lexical (multi-word reactions), not a speaker overfit.
- Next candidates (not run): a completion target separate from the timeout (Rin's redefinition):
  label each pause from the transcript as complete or incomplete and train a head whose target does
  not depend on how long the silence lasts, with the timeout left to the policy; onset-weighted
  loss on the first 0.1-0.4 s of each pause; pause-warp with synthetic room tone instead of copied
  frames.

## Reproduce

```bash
pip install 'modal[api-proxy-support]'
modal run ssl_turn/semantics/probe_eotbench.py --out probe_en.npz          # L4, ~5 min
python ssl_turn/semantics/probe_fit.py probe_en.npz [text_en.json]
modal run ssl_turn/semantics/dump_annotations.py --out ann.json            # raw labels, never commit
python ssl_turn/semantics/bc_check.py probs.npz ann.json                    # runs/r019_asr_bgaug/probs.npz
cd ssl_turn/pipeline && modal run pausewarp.py --groups 6
modal run train.py --run r020_warp --configs configs/r020_warp.json --n-train 131 --steps 1000 --gpu H100
```

eot-bench scoring of r020 uses `eotbench/modal_eot.py` and `to_harness.py` from draft PR #10.
`text_en.json` comes from eot-bench's `LiveKitTextTurnDetectorAdapter` (livekit/turn-detector
v0.4.1-intl) on the words before each pause plus the agent context.
