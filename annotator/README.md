# annotator: hindsight EOT / interruption labels from voice-activity timing

An offline labeler for two-speaker audio. It finds candidate points from per-speaker voice
activity, describes each by the timing of both speakers in a window from 3 s before to 4 s
after, and scores it with a gradient-boosted classifier trained on human labels
(TurnBench dev and otoSpeech). Content models also see what is said around each candidate:
streaming FastConformer encoder frames (`content.py`, the same encoder setup as
`ssl_turn/pipeline/encode_asr.py`) averaged over 1 s before to 1.5 s after, reduced by PCA.
Content matters most for interruptions; timing alone cannot tell them from other overlaps.

- Stereo input (one speaker per channel): Silero VAD per channel.
- Mono input (two speakers mixed): NVIDIA Sortformer 4spk v1 (offline, **CC-BY-NC-4.0**, so
  for non-commercial use only) in 4-min windows with 1-min overlap. Window slot orders are
  stitched by best permutation, and extra slots are folded into the two main speakers.
- Candidates: EOT at each speech offset followed by >= 0.2 s of silence. INT at each onset
  after >= 0.2 s of silence while the other speaker is talking, plus a point every 0.4 s
  (up to 3 s) inside each overlap. A label is placed at the candidate + 0.4 s (EOT) or
  + 0.1 s (INT).

```
python -m annotator.annotate --model labeler-content-stereo.joblib --mode stereo conv.wav --out labels/
python -m annotator.annotate --model labeler-content-mono.joblib --mode mono a.wav --precision 0.7 --out labels/
python -m annotator.train --data tbdev=DIR oto=DIR --eval          # timing-only cross-checks
python -m annotator.train --data tbdev=DIR oto=DIR --out model.joblib
modal run annotator/modal_content.py --data DIR --calibrate tbdev  # content models (Modal)
```
A content model needs the matching mode (stereo encodes each channel, mono the mix).
`--precision 0.7|0.8|0.9` uses thresholds where proposals reached that precision on held-out
TurnBench dev; the default threshold is TurnBench's FP <= 0.10 operating point, which proposes
far more interruptions than are real. The encoder takes about 2 min per 3 min of stereo audio
on 4 CPU cores (`--device cuda` for a GPU).
`annotate` writes `<stem>.json` (every candidate with its score, plus positive events) and
`<stem>.labels.txt`, an Audacity label track of the positive events (File > Import > Labels).
Using the track as a preprocessor for human labeling works well: at the stored thresholds a
reviewer sees about 5 proposals per real interruption and 2 per real turn end, and confirms or
rejects them instead of scanning the whole recording. Lower `--int-threshold` /
`--eot-threshold` to miss fewer events at the cost of more proposals.

Training data is `<dir>/<cid>.npz` (`silero_u8_32ms`: Silero probability x 255 per channel,
32 ms) plus `<dir>/<cid>.gold.json` (TurnBench-style events). Neither data nor model files are
committed; this project keeps them in the shared folder (`audio-llm/annotator-data/`,
`audio-llm/annotator-model/`).

Needs: numpy, scipy, scikit-learn, joblib, soundfile; onnxruntime + silero-vad (stereo);
nemo_toolkit[asr] (mono diarizer, and the encoder for content models); turnbench (pinned
scorer, for `--eval`). Content models are trained on Modal (`modal_content.py`) because the
encoder frames for the training data (`feats_asr` from ssl_turn) live on its volume.

## Results

All numbers are on held-out conversations. "Recall / FP" is TurnBench's official scorer at
FP <= 0.10. It only counts fires inside annotated negative spans as false, so it ignores fires
in unlabeled regions. "Recall@P" is the annotator view: every candidate outside excluded spans
counts, and an event counts as found when a candidate at or above the threshold falls in its
window. It reports event recall at the threshold where fired-candidate precision reaches P.

| task | test set | trained on | Recall / FP (TurnBench) | Recall@P0.7 | Recall@P0.8 |
|---|---|---|---|---|---|
| EOT | TB dev | TB dev (2-fold) | 0.870 / 0.100 | 0.789 | 0.647 |
| EOT | TB dev | otoSpeech | 0.893 / 0.100 | 0.751 | 0.571 |
| EOT | otoSpeech | otoSpeech (2-fold) | 0.926 / 0.098 | 0.886 | 0.809 |
| EOT | otoSpeech | TB dev | 0.915 / 0.098 | 0.891 | 0.787 |
| INT | TB dev | TB dev (2-fold) | 0.974 / 0.098 | 0.383 | 0.228 |
| INT | TB dev | otoSpeech | 0.991 / 0.099 | 0.023 | 0.000 |
| INT | otoSpeech | otoSpeech (2-fold) | 0.846 / 0.098 | 0.002 | 0.002 |
| INT | otoSpeech | TB dev | 0.801 / 0.099 | 0.007 | 0.007 |

Content features (stereo content model, same held-out protocol; mono = encoder on the mix):

| task | test set | trained on | timing only R@P0.7 / R@P0.8 | + content (stereo) | + content (mono mix) |
|---|---|---|---|---|---|
| INT | TB dev | TB dev (2-fold) | 0.383 / 0.228 | **0.709 / 0.524** | 0.646 / 0.427 |
| INT | TB dev | otoSpeech | 0.026 / 0.026 | **0.608 / 0.300** | 0.372 / 0.098 |
| EOT | TB dev | TB dev (2-fold) | 0.789 / 0.647 | 0.819 / 0.685 | 0.803 / 0.633 |

The shipped content models (`labeler-content-{stereo,mono}.joblib`) are trained on TB dev +
otoSpeech; their `--precision` thresholds come from cross-fitted scores on TB dev. There,
stereo reaches INT recall 0.778 at P0.7 and 0.548 at P0.8 (EOT 0.817 / 0.646); mono 0.651 /
0.406 (EOT 0.800 / 0.654).

On otoSpeech, interruption precision stays near zero with or without content. That looks like
its labels rather than the model: 38 of 119 conversations have no interruption labels at all,
and most top-scoring "false" interruptions are takeovers within 1 s of the other speaker's
turn end (inferred from the gold, not listened to). So interruption thresholds are calibrated
on TurnBench dev only (`--calibrate tbdev`). The mono numbers use stereo Silero timing; with
Sortformer timing on real mono audio expect somewhat lower.

**Timing alone does not give precise interruption labels.** On TurnBench's metric the labeler
looks excellent (INT recall 0.97-0.99 on TB dev). But most of its interruption fires land in
regions TurnBench leaves unlabeled, mostly overlaps that are not interruptions. As a source
of training labels its interruption events are only 15-25% precise. EOT labels are usable at
about 0.8 precision. Telling interruptions from other overlaps needs what was said and how,
not just when.

Mono vs stereo (21 otoSpeech conversations mixed to mono; TurnBench metric; earlier label
definition that ignored unlabeled regions): stereo INT 0.932 / EOT 0.941; Sortformer v1 + fold
0.945 / 0.919; streaming Sortformer v2.1 + fold 0.877 / 0.907.
