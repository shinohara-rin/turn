# Background speech (someone else talking near the user's mic)

Deployment problem: a TV, a YouTube video or another person talks in the room. That speech
lands only in the **user** channel, and the turn model treats it as the user: it fires false
interruptions while the agent talks, and misses or delays the user's end of turn.

## Test (`mixing.py`, `run_vap.py`, `score_all.py`; ssl_turn side in `ssl_turn/pipeline/bgmix.py`)
- 12 TurnBench dev conversations (seed 0, 2.17 h). One random channel per conversation is the
  "user". Its background is another TB dev conversation (both speakers, like a podcast playing),
  taken from the 26 conversations outside the subset, passed through a synthetic far-field
  RIR (RT60 0.3–0.7 s) and a 150–6000 Hz loudspeaker band-pass, at 10 / 5 / 0 dB relative to the
  user's annotated active speech.
- `gate5` / `gate0`: the same background but only while the user talks (an oracle target-speaker
  filter). This is the ceiling for any front end that removes non-user speech.
- Thresholds are picked on the clean audio (highest recall at FP ≤ 0.10), then held fixed.
- `false INT/min`: interruption fires per minute on a channel while that speaker is silent and
  the other one talks (the agent gets cut off). TurnBench only counts FPs inside annotated
  backchannel spans, so it misses most of these.

## Training fix (`bgmix.encode_aug`, `train.py` `bg_aug`)
Each otoSpeech train conversation gets one augmented copy per channel: a far-field podcast from
2 donor conversations outside train/dev, at SNR ~ U(-5, 20) dB, Cat-encoded
(`/work/feats_aug/oto`). With `bg_aug: 0.5`, half the training crops have one random channel
swapped for its augmented features; labels are unchanged. Run r015 (`configs/r015_bgaug.json`)
retrains the r012 head (fine1_bal1) with and without it, 2 seeds each, 1000 steps on 131 convs.

## Results (2026-10-09)

ssl_turn on the 12-conversation subset, at clean thresholds (mean of 2 seeds):

| condition | model | EOT recall / FP | user EOT recall | INT recall / FP | false INT/min (user) |
|---|---|---|---|---|---|
| clean | r015 baseline | 0.94 / 0.09 | 0.95 | 0.99 / 0.08 | 0.1 |
| clean | r015 bg_aug | 0.95 / 0.09 | 0.97 | 0.99 / 0.10 | 0.2 |
| 10 dB | baseline | 0.89 / 0.22 | 0.84 | 0.99 / 0.16 | 10.5 |
| 10 dB | bg_aug | 0.94 / 0.09 | 0.94 | 0.99 / 0.11 | 0.2 |
| 5 dB | baseline | 0.86 / 0.25 | 0.79 | 0.99 / 0.20 | 15.6 |
| 5 dB | bg_aug | 0.92 / 0.08 | 0.91 | 1.00 / 0.10 | 0.4 |
| 0 dB | baseline | 0.84 / 0.28 | 0.73 | 0.97 / 0.22 | 21.2 |
| 0 dB | bg_aug | 0.92 / 0.07 | 0.94 | 0.99 / 0.10 | 0.6 |

- Re-tuning thresholds on the noisy audio does not rescue the baseline (EOT recall 0.51–0.77).
- The oracle gate removes false interruptions, but the baseline still loses EOT FP (0.15) and INT
  FP (0.15–0.24) where background overlaps the user's own speech. Augmentation fixes both.
- Full clean TB dev (38 convs, official dev rule, 0.5 s refractory, EOT re-commit 1 s): bg_aug
  EOT 0.942 / 0.935 vs baseline 0.934 / 0.930; INT 0.974 / 0.977 vs 0.980 / 0.977. No clean cost.
- VAP (official oto checkpoint, 2 s refractory) on the same mixtures: user EOT 0.84 → 0.62 and
  false INT 0.2 → 7/min at 0 dB. It degrades less than the un-augmented ssl_turn head.
- Caveat: test background uses the same recipe as training (far-field dialogue playback), so this
  is in-distribution. Not yet tested: a near-field second talker (dry, loud), music or TV mixes,
  and real recordings.
