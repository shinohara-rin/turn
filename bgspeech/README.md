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
  is in-distribution. See the harder conditions below.

### Harder conditions (out of the augmentation's distribution)

`snrm5`: far-field playback at -5 dB. `near5` / `near0`: one other talker, dry (no room, no
loudspeaker), like someone talking right next to the mic. `tv5` / `tv0`: far-field dialogue with a
music bed 5 dB under it (MUSAN fma). `music0`: far-field music only. Mean of 2 seeds:

| condition | model | EOT recall / FP | user EOT recall | INT FP (user) | false INT/min (user) |
|---|---|---|---|---|---|
| -5 dB far | baseline | 0.80 / 0.33 | 0.66 | 0.44 | 26.6 |
| -5 dB far | bg_aug | 0.90 / 0.08 | 0.91 | 0.09 | 0.7 |
| TV 0 dB | baseline | 0.70 / 0.25 | 0.43 | 0.38 | 19.6 |
| TV 0 dB | bg_aug | 0.93 / 0.07 | 0.92 | 0.15 | 1.1 |
| music 0 dB | baseline | 0.75 / 0.11 | 0.53 | 0.13 | 2.5 |
| music 0 dB | bg_aug | 0.91 / 0.10 | 0.90 | 0.09 | 0.2 |
| near talker 5 dB | baseline | 0.87 / 0.20 | 0.82 | 0.29 | 16.2 |
| near talker 5 dB | bg_aug | 0.87 / 0.19 | 0.80 | 0.37 | 20.6 |
| near talker 0 dB | baseline | 0.85 / 0.23 | 0.77 | 0.33 | 18.7 |
| near talker 0 dB | bg_aug | 0.85 / 0.23 | 0.76 | 0.38 | 20.1 |

Augmentation generalizes to louder playback, TV with music and music alone, but does nothing for a
dry talker next to the mic. The model learned an acoustic cue (far-field, band-limited speech is not
the user), not who the user is. A near talker is acoustically just like the user, so separating them
needs speaker identity (an enrollment embedding) or spatial cues from a mic array.

### Near-talker attempts (r016, r017): negative

Both runs keep `bg_aug 0.5` but draw the augmentation from `feats_aug_near/oto` (a dry single
talker at SNR -5..15 dB, `bgmix.encode_aug(style="near")`). r016 `aug2` uses only that; r016 `spk`
and r017 `ecapa` also condition the head on a per-channel enrollment of the user's voice (FiLM plus
a frame-vs-enrollment similarity; `model.TurnModel.condition`). `spk` enrolls with the mean Cat
feature of 20 s of the speaker's active frames, `ecapa` with a SpeechBrain ECAPA speaker embedding
of the same 20 s (`bgmix.enroll_ecapa`). Full clean TB dev (official dev rule, seeds s1 / s2):

| model | EOT recall | INT recall |
|---|---|---|
| r015 bg_aug (far-field only) | 0.942 / 0.935 | 0.974 / 0.977 |
| r016 aug2 (near aug) | 0.847 / 0.889 | 0.767 / 0.839 |
| r016 spk (near aug + Cat-mean enrollment) | 0.868 / 0.818 | 0.853 / 0.810 |
| r017 ecapa (near aug + ECAPA enrollment) | 0.800 / 0.861 | 0.648 / 0.744 |

On the 12-conversation mixtures, user-channel false INT/min at near talker 0 dB drops from ~20
(r015) to 2.9 / 4.6 (aug2), 2.5 / 2.5 (spk) and 3.5 / 5.9 (ecapa). But every near-aug model loses
0.05-0.13 EOT and 0.1-0.3 INT recall on clean audio, and they score *better* on far-field mixtures
than on clean (e.g. ecapa_s1 EOT 0.81 clean vs 0.87 at 0 dB): training on a voice that sounds
exactly like the user but must be ignored teaches the head to distrust speech evidence in general.
Neither enrollment fixes that; a real speaker-verification embedding (ECAPA) is no better than the
Cat mean. Recommendation stays r015. A near talker likely needs either an explicit target-speaker
front end (personalized VAD / TSE trained on separate data) or a mic-array spatial cue, not more of
this augmentation.
