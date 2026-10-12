# EOT latency: head objective vs. training data (2026-10-11)

Goal (Rin): cut end-of-turn latency. Starting point: r019 (`r019_asr_bgaug`, FastConformer + floor
head) answers in ~800 ms at 5% false cutoffs on LiveKit eot-bench en, and its score climbs with
silence (pause-onset AUC 0.85 at 0.1 s) while an encoder probe fit *on eot-bench* reaches 0.91
(`semantics/README.md`). Two hypotheses: the head's objective under-uses the encoder at pause
onset, or the training data (two-party chat, otoSpeech) is the wrong shape for voice-agent turns.

All numbers: eot-bench en (400 user turns, silent agent channel, LiveKit's harness), mean latency on
true ends at 2 / 5 / 10% false-cutoff budgets, cutoff rate at 300 ms mean latency, and AUC of ends
vs holds at a fixed silence. TB dev uses the commit policy (`policy.py`) at its default settings.

## 1. Head objective (r023, `pipeline/configs/r023_onset.json`): no gain

Six arms on the r019 recipe, two seeds each, trained in lockstep (H100, 1000 steps):
- `yld`: a per-speaker "this pause yields" head (`labels.pause_targets`): one target per pause, fixed
  at its onset from the gold floor, so silence length is never part of the label; the first 0.4 s
  weighs 4x, supervised up to 2 s.
- `yldon`: the same head supervised only on the first 0.4 s of each pause.
- `yldmute`: `yld` plus `mute 0.3`: one channel of 30% of crops replaced with the FastConformer
  encoding of digital silence (eot-bench's agent channel, `silent_feats.py`); only the other
  speaker's own targets stay supervised.
- `onfl`: floor loss x4 on the first 0.4 s of every pause. `onfl_yldmute`: both.

| eot_q unless noted | AUC | 2% | 5% | 10% | cutoff @300 ms | onset AUC 0.1 s / 0.3 s |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ctrl s1 / s2 (r019 recipe) | 0.976 / 0.975 | 1583 / 1622 | 789 / 769 | 486 / 509 | 19.9 / 21.1% | 0.853 / 0.903 |
| yld s1 / s2 | 0.974 / 0.969 | 1558 / 1652 | 901 / 828 | 546 / 577 | 21.0 / 23.4% | 0.846 / 0.896 |
| yldon s1 / s2, `yld_q` | 0.949 / 0.945 | 1594 / 1446 | 846 / 854 | 590 / 595 | 21.7 / 21.1% | **0.874** / 0.911 |
| yldmute s1 / s2 | 0.979 / 0.974 | 1599 / 1561 | 754 / 855 | 501 / 581 | 21.8 / 24.1% | 0.836 / 0.891 |
| onfl s1 / s2 | 0.970 / 0.965 | 1741 / 1728 | 855 / 846 | 558 / 590 | 20.9 / 23.4% | 0.868 / 0.907 |
| onfl_yldmute s1 / s2 | 0.978 / 0.975 | 1434 / 1456 | 770 / 802 | 522 / 564 | 21.1 / 22.1% | 0.853 / 0.900 |

Seed spread is ~20-90 ms; no arm beats the control at 5% or 10%. The onset-only yield head lifts
pause-onset AUC a little (0.853 -> 0.874) but not latency. The objective was not the limit.

## 2. Voice-agent turns as training data (r024, r025): -150 ms

`eotb_data.py` turns eot-bench's *other* languages (de, fr, it, nl, pt, ar, hi, id, ko, tr; 3,954
user turns, 11 h, CC-BY-4.0) into training sessions: user audio in channel 0 through the same
FastConformer, the agent channel the encoding of silence, labels from the dataset's silence spans
(holds keep the floor, the final span opens it; agent channel, fine labels and VAP unsupervised).
English (reported here) and es / ja / zh (the multilingual thread's benchmarks) are never used.
`train.py --eotb_aug p` draws a crop from these sessions with probability p, otherwise otoSpeech.

| eot_q unless noted | AUC | 2% | 5% | 10% | cutoff @300 ms | onset AUC 0.1 / 0.3 / 1.0 s |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ctrl (r023) mean of 2 seeds | 0.976 | 1602 | 779 | 497 | 20.5% | 0.853 / 0.903 / 0.895 |
| r024 eotb 0.3, s1 / s2 | 0.981 / 0.980 | 1269 / 1267 | 631 / 717 | 375 / 397 | 14.6 / 16.6% | 0.857 / 0.929 / 0.923 |
| r024 eotb 0.3 + yldon s1 / s2 | 0.983 / 0.979 | 1196 / 1387 | 689 / 679 | 409 / 402 | 13.9 / 15.7% | 0.856 / 0.930 / 0.920 |
| r024 same, `yld_q` | 0.976 / 0.968 | 1096 / 1242 | 693 / 724 | 421 / 455 | 14.3 / 15.7% | **0.894** / 0.945 / 0.914 |
| r024 eotb 0.3 + mute s1 / s2 | 0.980 / 0.980 | 1458 / 1519 | 640 / 685 | 401 / 396 | 15.3 / 16.0% | 0.858 / 0.925 / 0.919 |
| **r025 eotb 0.5, s1 / s2** | 0.980 / 0.982 | 1209 / 1132 | **643 / 612** | **386 / 385** | 14.9 / 14.3% | 0.851 / 0.930 / 0.925 |
| r025 eotb 0.5 + yldon s1 / s2 | 0.980 / 0.981 | 1242 / 1214 | 697 / 634 | 407 / 424 | 15.3 / 14.6% | 0.846 / 0.928 / 0.912 |

For reference (LiveKit's committed runs): LiveKit v1 543 / 295 ms, Baton 577 / 350, Soniox 647 / 512
at 5% / 10%. r025 eotb5 s2 at 612 / 385 would move from 4th to 3rd of 13.

TB dev at the default policy is unchanged (EOT R / FP / p50, INT R / FP / p50):

| | EOT | INT |
| --- | --- | --- |
| r023 ctrl s1 / s2 | 0.978 / 0.081 / 505, 0.978 / 0.085 / 480 | 1.000 / 0.087 / 445, 0.997 / 0.099 / 440 |
| r024 eotb s1 / s2 | 0.978 / 0.083 / 514, 0.978 / 0.083 / 488 | 0.994 / 0.082 / 434, 0.994 / 0.087 / 435 |
| r024 eotb + yldon s1, `eot_q` / `yld_q` trigger | 0.978 / 0.083 / 486, 0.978 / 0.082 / **454** | 0.997 / 0.087 / 434 |
| r025 eotb5 s1 / s2 | 0.978 / 0.085 / 502, 0.977 / 0.080 / 527 | 0.994 / 0.077 / 453, 0.994 / 0.080 / 455 |

- **The latency gap was data, not the objective.** 11 h of voice-agent turns, in languages the
  English encoder was not trained on, cut eot-bench en latency by ~150 ms at 5% and ~110 ms at 10%,
  and the 300 ms cutoff rate from ~20% to ~15%. Holds still climb with silence (hold 0.33 at 1.0 s),
  but ends now separate from holds by 0.3 s (AUC 0.93 vs 0.90).
- Mixing 0.3 or 0.5 of crops gives about the same; the source (3,954 turns) is probably what limits
  it. More voice-agent-style English is the obvious next lever.
- The onset-only yield head (`yld_q`) is the best pause-onset detector we have (AUC 0.89-0.90 at
  0.1 s, close to the encoder probe's in-domain 0.91) and, as the TB policy's EOT trigger, cuts
  median EOT latency 486 -> 454 ms at equal recall and FP. On eot-bench its operating points are
  no better than eot_q's, because it stays flat on long holds where eot_q's climb helps the timeout.

## Reproduce

```bash
pip install 'modal[api-proxy-support]'
modal run ssl_turn/latency/silent_feats.py                 # silent agent-channel features (mute arm)
modal run ssl_turn/latency/eotb_data.py                    # voice-agent sessions (L4 x 10, ~$0.3)
cd ssl_turn/pipeline
modal run train.py --run r023_onset --configs configs/r023_onset.json --n-train 131 --steps 1000 --gpu H100
modal run train.py --run r025_eotb5 --configs configs/r025_eotb5.json --n-train 131 --steps 1000 --gpu H100
cd ../.. && modal run eotbench/modal_eot.py --run r025_eotb5 --models eotb5_s1,eotb5_s2 --out tracks.npz
python eotbench/to_harness.py tracks.npz en --variant eot_q --tag r025   # then eot-harness (eotbench/README.md)
modal run ssl_turn/pipeline/policy_score.py::main --run r025_eotb5 --model eotb5_s2 [--eot-var yld_q]
```

Modal cost for this part: about $1.5 (training $1.0, data build $0.3, eot-bench scoring $0.1).
