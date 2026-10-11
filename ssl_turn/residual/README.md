# Residual model on top of VAD + rules (2026-10-11)

Question (Rin): instead of a full turn model, train a model only on the cases where Silero VAD +
the commit policy (`../policy.py`) is wrong or late?

Everything here runs on CPU from caches on the Modal volume `ssl-turn-work` (`data.py` lists the
`modal volume get` commands; no Modal compute was used). eot-bench English was re-encoded locally
with the same FastConformer code (`encode_eotb.py`, ~16 min on 4 CPU cores).

## 1. Where VAD + rules goes wrong on TB dev (`analyze.py`)

Rules at the FP 0.10 operating point (`vad_th 0.5, min_wait 0.6, other_dur 0.3`):
R 0.984 / FP 0.091 / p50 616 ms. Every fire traced to its VAD offset (candidate) and trigger:

| | candidates | TP | FP | lands in unscored audio |
|---|---:|---:|---:|---:|
| first fire: other speaker started | 6,780 | 1,752 | 88 | 4,940 |
| first fire: 2.5 s deadline | 156 | 66 | 8 | 82 |
| confirmation fire (+2.5 s) | 3,736 | 100 | 14 | |
| cancelled by own resumption | 4,371 | | | |

31 positives get no fire at all. So TB EOT is mostly "the other person started talking": 96% of
TPs come from that trigger, at a median 0.60 s after the offset. The rules' residuals are
(a) ~97 false fires, almost all from the other speaker starting during a hold (backchannels,
early starts), and (b) latency.

Oracle residuals (gold-informed upper bounds on the rules policy):

| oracle | R / FP / p50 |
|---|---|
| rules | 0.984 / 0.091 / 616 |
| veto every false candidate | 0.984 / 0.001 / 616 |
| fire every true candidate at offset + 0.32 s | 0.977 / 0.094 / 305 |
| both | 0.977 / 0.004 / 305 |
| r019 + rules today (`policy.py`, budget 0.10) | 0.985 / 0.100 / 520 |

r019 fires 64% of the other-triggered TPs earlier, by a median 0.12 s. So on TB the model's edge is
~100 ms and the ability to reach FP 0.07-0.08; the oracle says a perfect candidate classifier
could reach ~300 ms at near-zero FP, so the headroom is in classifying candidates, which is what a
residual model does.

## 2. Residual model: a head trained only on VAD candidates (`cands.py`, `train_res.py`)

Rows: every VAD offset (vad_th 0.5, gaps < 0.2 s merged) probed every 0.1 s up to 1.5 s, then
2.0 / 2.5 s, while the speaker is still silent. Label 1 if a gold EOT anchor sits within
[-0.3, +0.8] s of the offset (anchors sit at the VAD offset: median +0.04 s), 0 if a fire at the
probe would be a scored false positive, else unused. Features known at the probe: FastConformer
frame of the speaker and of the other channel, mean of the speaker's last 1 s of frames, and VAD
timing scalars. Model: 1-hidden-layer MLP (128-256 units). Training sets: otoSpeech train (131
convs, 165k labeled rows) and, for res3/res4, the 11 h of eot-bench voice-agent turns in other
languages that r025 used (`cands.py eotb`).

Candidate AUC (labeled rows, all probes):

| | oto dev | TB dev | TB dev @0.1 s | TB dev @0.3 s |
|---|---:|---:|---:|---:|
| r019 eot_q (full head, oto) | 0.838 | 0.901 | 0.790 | 0.848 |
| r025 eot_q (full head, oto + voice-agent) | 0.835 | 0.897 | 0.785 | 0.837 |
| VAD timing scalars only | 0.75 | 0.83 | | |
| res2 (oto) | 0.851 | 0.905 | 0.837 | 0.845 |
| res4 (oto + voice-agent, lr 2e-4) | 0.857 | 0.918 | 0.847 | 0.861 |

The residual head is better than the full head early in a pause (+0.05 AUC at 0.1 s) and equal
or slightly better later. It overfits after one epoch.

## 3. Through the TB dev policy (`policy_res.py`)

Residual score as an early trigger (fire at the first probe >= d_min with score >= th) and as a
veto (other-speaker trigger waits for the deadline when the score is < lo). Held-out = settings
picked on 19 conversations, scored on the other 19, 200 splits.

| FP budget 0.10 | dev R / FP / p50 | held-out R / FP / p50 |
|---|---|---|
| rules | 0.984 / 0.091 / 616 | 0.984 / 0.095 / 618 |
| r019 + rules (`policy.py`) | 0.985 / 0.100 / 520 | 0.985 / 0.105 / 525 |
| res4 early | 0.985 / 0.098 / 491 | 0.985 / 0.103 / 493 |
| res4 early + veto | 0.982 / 0.098 / 525 | 0.981 / 0.098 / 461 |
| **FP budget 0.08** | | |
| r019 + rules (`policy.py` default, vad_th 0.7) | 0.977 / 0.072 / 532 | 0.978 / 0.088 / – |
| res4 early + veto | 0.978 / 0.080 / 525 | 0.978 / 0.084 / 515 |

About 30 ms faster than r019 at the same recall and FP, which is inside split-to-split noise
(res2 and res3 land between 475 and 555 ms held-out). The veto gives no clear gain.

## 4. eot-bench en (`encode_eotb.py`, `eotb_res.py`, `eotb_heads.py`)

Each silence span is a candidate; the residual reads the same features at each 0.1 s grid point
(agent channel silent). Scored with LiveKit's harness (`eot-harness compute-metrics`). The CPU
re-runs of the full heads reproduce the Modal runs (r019 826 vs 815 ms, r025 621 vs 612 ms @5%).

| | AUC | 2% cutoff | 5% cutoff | 10% cutoff |
|---|---:|---:|---:|---:|
| VAD baseline (harness) | | | 1600 ms | 1000 ms |
| r019 (full head, oto) | 0.967 | 1607 | 826 | 575 |
| r025 (full head, oto + voice-agent) | 0.980 | 1261 | 621 | 387 |
| res2 (oto) | 0.882 | 1718 | 1127 | 772 |
| res3 (oto + voice-agent) | 0.970 | 1047 | 768 | 501 |
| res4 (oto + voice-agent, lr 2e-4) | 0.973 | 1185 | 771 | 508 |
| mean(r025, res3) | 0.982 | 1070 | 691 | 366 |
| mean(r025, res4) | 0.984 | 1122 | 679 | 378 |

Trained on oto alone the residual transfers badly to voice-agent pauses (1127 ms). With the
voice-agent turns it is ~150 ms behind the full head trained on the same data at 5%, better at 2%;
blending helps at 2% and 10% but costs ~60 ms at 5%.

## Reading

- Training only on VAD candidates works: the residual head matches the full head on TB dev and
  separates ends from holds better in the first 0.1-0.3 s of a pause. It does not move the
  operating points: TB dev p50 ~490-520 ms either way (oracle bound ~300 ms), and on eot-bench it
  is behind r025 at the 5% budget.
- The model's edge over VAD + rules: TB EOT ~100 ms and FP 0.09 -> 0.07; TB INT +2.3 pts recall
  at FP 0.05 and ~500 ms earlier (rules 0.963 / 0.050 / 1133 ms vs 0.986 / 0.046 / 554 ms); on
  eot-bench ~1 s at 5% cutoffs (1600 -> 621 ms).
- The residual head still needs the 114M FastConformer; with VAD timing alone the candidate AUC
  drops to 0.83 (TB). Making a residual model small is the distillation problem again
  (`../distill`, PR #17), where students fail on eot-bench.

## Reproduce

```
python analyze.py                                   # section 1
python cands.py && python cands.py eotb             # candidate rows -> $DATA/cands
python train_res.py --tag res2                      # oto only (hidden 256, lr 1e-3)
python train_res.py --tag res3 --extra eotb_train
python train_res.py --tag res4 --extra eotb_train --lr 2e-4 --hidden 128 --epochs 6
python policy_res.py --tag res4                     # section 3
python encode_eotb.py                               # eot-bench en features, CPU
python eotb_res.py --tags res2,res3,res4
python eotb_heads.py --heads r019_bgaug_s1,r025_eotb5_s1 --blend res3   # needs the .pt heads in $DATA
for d in $DATA/eotb/*; do eot-harness compute-metrics --predictions $d/predictions.parquet --output-dir $d/metrics; done
```

The harness needs its own venv (`pip install git+https://github.com/livekit/eot-bench pyarrow==18.1.0 "pandas<3"`).
Result files (harness predictions + metrics, logs) are in the project folder `residual/`.

## 5. Pause-focused distillation (d004, `../distill`, RTX 3090)

Follow-up: if the pauses carry the signal, a small student trained with more weight on them
might close its eot-bench gap to the teacher. `distill.py` gains `pause_w`. `pause_frames` marks
frames where a speaker the teacher calls SILENT was talking within the last 2.5 s (the student's
analogue of a VAD candidate). The floor / future / yld KD terms are a weighted mean with weight
1 + pause_w there. d004 is d002's recipe (teacher r025 eotb5_s2, 155 oto conversations + 30%
voice-agent rows, 6000 steps): three s3m students on shared batches, pause_w 0 / 3 / 8. It was
trained on the Ray 3090 (`distill/node.py`, 585 ms/step, 5.9 GB peak) and scored on CPU here.
The local eot-bench pipeline reproduces d002 s3m (`yld_q` 1057 vs 1060 ms @5%).

| eot-bench en | AUC | 2% | 5% | 10% |
|---|---:|---:|---:|---:|
| teacher r025 | 0.980 | 1261 | 621 | 387 |
| d002 s3m `yld_q` (H100) | 0.936 | 2166 | 1057 | 705 |
| d004 s3m control `yld_q` | 0.939 | 1971 | 1095 | 728 |
| d004 s3m pause_w 3 `yld_q` | 0.924 | 1891 | 1193 | 755 |
| d004 s3m pause_w 8 `yld_q` | 0.922 | 2051 | 1158 | 740 |
| d004 control / p3 / p8 `eot_q` | 0.928 / 0.924 / 0.923 | 2119 / 1947 / 2066 | 1208 / 1300 / 1244 | 772 / 800 / 749 |

NEGATIVE: up-weighting pauses makes the students slightly worse on eot-bench (AUC -0.015, +60-100
ms at 5%), though the weighted students reach a lower oto-dev KD loss (0.705 vs 0.709). Where the
loss falls doesn't seem to be what limits the students; it's what the small encoder can represent.

TB dev through the policy (`colab.py sweep d004 s3m,s3m_p3,s3m_p8`; the sweep picks the best model
per family): EOT 0.978 / 0.078 / 550 ms at FP 0.08 (held-out 0.978 / 0.085), INT 0.997 / 0.080 /
584 ms at 0.10 (held-out 0.993 / 0.098). That is teacher level as before, and the picks are spread
across all three students. So TB dev doesn't separate them; the difference shows only on eot-bench.
