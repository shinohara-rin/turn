# Distilling the pipeline into a small student (2026-10-11)

Rin's question: can the deployed pipeline (Silero VAD + streaming FastConformer 114M + floor head
3M + the commit policy) become a much smaller model with the same performance? The VAD (~2 MB) and
the policy (rules, no parameters) are already small, so the student replaces the two model parts:
FastConformer + head become one causal network that reads 16 kHz stereo audio (`student.py`):

- fixed-weight log-mel front end (25 ms Hann, 10 ms hop, 80 bins) as a strided conv, so ONNX and
  streaming need no STFT op;
- per channel: a causal conv over 10 ms frames, 8 frames stacked into one 80 ms frame, then
  `layers` blocks of depthwise causal conv + windowed causal attention (5.6 s) + feed-forward;
- the teacher's own floor head (`model.TurnModel`: cross-channel attention, every output head).

| preset | inference params | encoder | fp32 ONNX | CPU, 1 thread, per 80 ms frame (K=1 / 4) |
| --- | ---: | ---: | ---: | --- |
| s1m | 1.71M | 1.20M | 7.3 MB | 2.2 ms (RTF 0.028) / RTF 0.009 |
| s3m | 3.41M | 2.52M | 13.7 MB | 3.2 ms (RTF 0.040) / RTF 0.012 |
| s8m | 9.15M | 6.32M | 35.6 MB | 6.2 ms (RTF 0.077) / RTF 0.030 |
| teacher (FastConformer + head) | 117M | 114M | 436 MB | 64.6 ms (RTF 0.81) |

Timings: onnxruntime on this container's CPU (`bench_onnx.py`), teacher = the browser export's
`encoder.onnx` + `head.onnx` from PR #11 on the same machine. `stream_student.StudentStep` is one
graph with fixed-size caches and reproduces the offline forward exactly (`test_student.py`).

## Training (`distill.py`)

Teacher logits are precomputed by exact causal chunked inference over the cached FastConformer
features (`teacher/{tag}`). Students see random 30 s crops of 16 kHz audio and are trained on:
soft cross-entropy to the teacher's floor, projections, acts, fine labels and VAP (BCE for yld);
gold labels (`model.loss`, weight 0.5); MSE to the teacher's standardized 1024-d FastConformer
frame through a linear map (weight 1, training only). Half the crops take one channel from a
background-augmented copy (r019's recipe: far .5 / tv .2 / babble .2 / music .1, SNR -5..20 dB)
while every target stays the clean teacher's. Swap and per-channel gain augmentation. 6000 steps,
batch 48, H100, best checkpoint by oto-dev floor + future KD loss (all peak at 2500-4000 steps).

- **d001**: teacher r019 `bgaug_s2`; data = the 131 otoSpeech train conversations (32 h).
- **d002**: teacher r025 `eotb5_s2` (the latency-tuned head, `latency/README.md`); data = 155 oto
  conversations (train + the gate/donor ones outside dev) plus 30% of crops from the 11 h of
  voice-agent turns in eot-bench's other languages.

## Results

TurnBench dev through the commit policy, settings re-selected per model on the same grid
(`policy_score.py::sweep`; dev R / FP / p50 ms, then split-half held-out R / FP):

| model | EOT, FP <= 0.08 | EOT, FP <= 0.10 | INT, FP <= 0.10 |
| --- | --- | --- | --- |
| teacher r019 bgaug_s2 | 0.978 / 0.080 / 580; 0.977 / 0.088 | 0.984 / 0.100 / 563; 0.981 / 0.105 | 0.997 / 0.099 / 453; 0.991 / 0.096 |
| d001 s8m | 0.979 / 0.074 / 663; 0.977 / 0.085 | 0.984 / 0.096 / 570; 0.982 / 0.103 | 1.000 / 0.096 / 392; 0.994 / 0.097 |
| d001 s3m | 0.981 / 0.079 / 553; 0.978 / 0.086 | 0.984 / 0.096 / 519; 0.982 / 0.103 | 1.000 / 0.098 / 500; 0.994 / 0.094 |
| d001 s1m | 0.979 / 0.079 / 552; 0.978 / 0.084 | 0.985 / 0.095 / 573; 0.982 / 0.104 | 0.994 / 0.095 / 514; 0.987 / 0.096 |
| teacher r025 eotb5_s2 | 0.979 / 0.080 / 546; 0.977 / 0.086 | 0.984 / 0.099 / 518; 0.982 / 0.103 | 0.994 / 0.094 / 455; 0.987 / 0.097 |
| d002 s8m | 0.978 / 0.078 / 596; 0.977 / 0.087 | 0.984 / 0.098 / 503; 0.981 / 0.105 | 0.994 / 0.099 / 423; 0.989 / 0.097 |
| d002 s3m | 0.979 / 0.078 / 562; 0.977 / 0.084 | 0.984 / 0.099 / 557; 0.982 / 0.102 | 0.997 / 0.099 / 424; 0.991 / 0.097 |
| d002 s1m | 0.979 / 0.074 / 673; 0.977 / 0.086 | 0.984 / 0.097 / 571; 0.982 / 0.104 | 0.994 / 0.100 / 387; 0.988 / 0.098 |

bgbench (38 TB dev conversations, background in the user channel, each model at its clean
operating point; d001 students, r019 `bgaug_s2` as the teacher):

| model | clean EOT / INT | playback EOT drop, false INT/min | music | near talker | babble |
| --- | --- | --- | --- | --- | --- |
| teacher | 0.936 / 0.986 | -0.023, 1.1 | -0.009, 0.4 | -0.081, 8.4 | -0.057, 1.8 |
| d001 s8m | 0.939 / 0.983 | -0.073, 1.9 | -0.059, 0.6 | -0.158, 9.9 | -0.118, 3.6 |
| d001 s3m | 0.934 / 0.983 | -0.066, 1.8 | -0.070, 0.4 | -0.175, 10.3 | -0.094, 3.2 |
| d001 s1m | 0.941 / 0.983 | -0.079, 4.0 | -0.050, 0.9 | -0.172, 11.4 | -0.134, 7.9 |

LiveKit eot-bench en (silent agent channel; mean latency at 2 / 5 / 10% cutoff, cutoff @300 ms,
pause-onset AUC at 0.1 / 0.3 s):

| model | AUC | 2% | 5% | 10% | cutoff @300 ms | onset AUC |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| teacher r019 bgaug_s2 | 0.975 | 1660 | 805 | 539 | 21.0% | 0.85 / 0.90 |
| d001 s3m / s8m | 0.941 / 0.925 | 1820 / 1865 | 1126 / 1170 | 781 / 801 | 41.4 / 41.7% | 0.74 / 0.80 |
| teacher r025 eotb5_s2 | 0.982 | 1132 | 612 | 385 | 14.3% | 0.85 / 0.93 |
| d002 s3m `yld_q` | 0.936 | 2169 | 1060 | 705 | 22.6% | 0.82 / 0.88 |
| d002 s1m `yld_q` | 0.933 | 2062 | 1001 | 654 | 27.0% | 0.81 / 0.86 |
| d002 s8m `eot_q` | 0.909 | 2005 | 1191 | 759 | 26.5% | 0.76 / 0.86 |

## Verdict

- **On TurnBench the pipeline distills.** A 3.4M student (1/34 of the teacher's parameters, 20x
  faster on one CPU thread, 14 MB) matches the teacher on TB dev through the commit policy: EOT
  0.978-0.981 at FP 0.08, INT 0.994-1.000 at FP 0.10, held-out within 0.004 of the teacher, at
  similar or lower median latency. Even 1.7M holds up. Clean bgbench is also on par.
- **It does not inherit the encoder's robustness.** With background audio the students lose
  2-3x more EOT recall than the teacher and fire 1.5-4x more false interruptions (babble worst).
- **Nor its generalization to new audio.** On eot-bench (16 kHz voice-agent recordings, a domain
  none of the oto training audio covers) d001 is ~1.1-1.2 s at 5% vs the teacher's 0.8 s. Adding
  the voice-agent turns to the student's data (d002) helps (cutoff @300 ms 41% -> 23%, 5% latency
  -45 to -225 ms) but stays far from its teacher (612 ms). The 114M encoder's ASR pretraining on
  thousands of hours carries this; ~45 h of audio does not transfer it.
- Size barely matters between 1.7M and 9.2M on any benchmark; data does. Next lever, untested:
  distill on a large unlabeled audio pool (teacher pseudo-labels need only the FastConformer
  pass, ~$0.8 per 1000 channel-hours on an L4), with background mixing, before any architecture
  change.
- Feature distillation (`feat`) made no measurable difference at this data size (d001 s3m vs
  s3m_nofeat: TB dev and eot-bench within seed noise).

## Reproduce

```bash
cd ssl_turn/distill
modal run distill.py::prep                                             # 16 kHz audio + bg copies
modal run distill.py::prep_eotb                                        # voice-agent sessions (after latency/eotb_data.py)
modal run distill.py::teacher --run r025_eotb5 --name eotb5_s2 --tag r025s2 --splits oto,tbdev,eotb
modal run distill.py::train --run d002 --configs configs/d002.json --teacher r025s2 --extra
modal run ../pipeline/policy_score.py::sweep --run d002 --models s3m
modal run distill.py::eot --run d002 --names s3m --out tracks.npz       # then eotbench/to_harness.py
cd ../../bgspeech && modal run bench_modal.py::student --run d001 --members s3m
python ssl_turn/distill/bench_onnx.py --out /tmp/onnx                   # CPU speed
```

Modal cost for this part: about $5.6 (data prep, teacher dumps and two student runs $4.5, bgbench $1.1), plus $0.75 of CPU policy sweeps.
