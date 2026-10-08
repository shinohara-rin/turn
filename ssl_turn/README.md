# Attempt 2: large-SSL causal backbone + stereo turn model

A second TurnBench attempt, running in parallel with the Ooma-style recreation in
`training/` (Parakeet 120M + VAP continuation + MLP head). The two attempts share the
data protocol but are otherwise independent.

Target: Ooma Turn Detector, test EOT 0.951 / FPR 0.082 / p50 388 ms and INT 0.968 /
0.076 / 567 ms (leaderboard #1 is Vox Maru v1 at EOT 0.960). The thesis is that the
largest gain still available is **representation scale**. The plan inherits web-scale
pretraining from an existing causal encoder instead of doing SSL from scratch, then
spends our own compute on dyadic (two-channel) objectives and on more labeled data.

## Backbone: MOSS-Audio-Tokenizer ("Cat") encoder, not the MOSS-Audio encoder

Both come from OpenMOSS and are Apache-2.0. Only one is usable causally.

| | MOSS-Audio encoder (in the 4B/8B audio LLMs) | MOSS-Audio-Tokenizer encoder (Cat) |
|---|---|---|
| Pretraining | Trained from scratch inside an audio-understanding LLM; data scale not stated | ~3M h of speech, sound and music: reconstruction plus LLM semantic alignment, from scratch |
| Size | 32L, d1280, ~0.64B | 3×12L d768 (100/50/25 Hz) + 32L d1280 (12.5 Hz); **0.887B** |
| Attention | **Bidirectional within independent 4 s chunks** (`n_window=200`) | **Causal**, 10 s window, non-overlapping patches |
| Front end | Whisper log-mel (per-clip max normalization, centered STFT) | Raw 24 kHz waveform, no STFT |
| Rate | 12.5 Hz | 12.5 Hz (80 ms per frame, no lookahead) |
| Causal use | Re-encode a past window every step, out of distribution, or retrain the mask | Native streaming with KV cache |

Pins: `OpenMOSS-Team/MOSS-Audio-Tokenizer@3cd226ba2947efa357ef453bcad111b6eafba782`;
code SHA256 is in `cat_encoder.py`. The encoder-only weight extract
(685 tensors, 3.55 GB fp32) has SHA256
`c04f98495f46d9a709cbfd4dbd122dde90b9a932c4075e00505f095946aa4993`.

The MOSS-Audio LLM is still useful, but **offline as a labeler**: it does audio captioning
and event localization. See "Pseudo-labels" below.

### Verified so far (CPU, this container, 2026-10-08)
`python -m unittest test_cat_encoder test_model`, with `CAT_DIR` set for real weights:
- **Exactly causal:** with real weights on 12.8 s of stereo audio, randomizing everything
  after frame 140 leaves frames 0–139 unchanged (atol 1e-5). Frame *k* becomes available at
  `(k+1)·80 ms` plus the resampler delay.
- **Matches upstream:** our encoder-only build (upstream classes, no quantizer or decoder) is
  bit-identical to upstream `MossAudioTokenizerModel._encode_frame(...).encoder_hidden_states`
  (shrunken random config).
- **Found and fixed an upstream streaming bug:** upstream's ring KV cache holds exactly
  `context` keys. A chunk of T tokens therefore evicts keys that its earliest queries still
  need, so upstream chunked `encode()` drifts from the full-sequence result past 10 s, which
  covers every conversation. `CatEncoder.stream()` enlarges the caches; the position mask
  already enforces the window. Streaming now equals the full sequence for any chunk size, on
  real weights past the window as well.
- **Throughput** (4 vCPU, fp32, stereo): offline encoding runs at 0.65× real time; 160 ms
  streaming chunks run at 2.6× real time, i.e. slower than real time. Feature caching on a
  GPU is therefore cheap: otoSpeech's 104 h × 2 channels is roughly 35 PFLOP. A browser
  build would need distillation (stage 4).

## Model (`model.py`)

- Inputs per speaker channel: a softmax-weighted mix of top-stage taps (layers 8/16/24/32,
  d1280) plus the final 768-d output, each LayerNorm→Linear to d=256, plus a channel embedding.
- Four layers of (shared per-channel causal self-attention → causal cross-attention to the
  other channel), as in VAP's stereo transformer. ALiBi recency bias with a hard 20 s window
  gives the same receptive field when training on crops and when streaming.
- Heads:
  - joint **256-class VAP** (bins 0.24/0.40/0.56/0.80 s over a 2 s horizon);
  - per-speaker **EOT**, **INT** and **VAD** logits from `[own, other]`, using shared
    weights so swapping speakers swaps the outputs (tested).
- Decisions every 80 ms rather than attempt 1's 160 ms grid, which removes up to 80 ms of
  quantization latency.

Unlike attempt 1, INT gets a head trained for it from the start.

## Training stages

All stages run remotely (Colab/Modal GPU), as in `training/README.md`. Nothing here
downloads datasets locally.

0. **Frozen probe.** Cache Cat taps for otoSpeech and train `TurnModel` with VAP+EOT+INT+VAD
   on the frozen split's human labels, reusing `training/turn_detector`'s `annotations`,
   `gold_and_activity` and `eot_targets`, plus `leakage_guard`. Sweep tap layers on a subset
   first, because all four taps cost about 77 GB fp16 for 104 h versus 14 GB for the final
   layer alone; then cache only the chosen layers. Score on TurnBench dev with the pinned
   evaluator. **This is the first go/no-go:** Cat features have to beat frozen Parakeet under
   the same head budget.
1. **Pseudo-labeled scale-up.** See the next section.
2. **Encoder adaptation.** LoRA on, or unfreezing of, the top N Cat layers with the VAP
   objective, which needs only per-channel activity. It therefore scales to any separated
   two-channel conversation audio, labeled or not. The encoder stays causal under
   fine-tuning because the mask is structural.
3. **Operating point.** The quiet-gated commit policy and threshold selection follow the
   pinned `turnbench.sweep` on dev. A learned VAD head and Silero together serve as the gate.
4. **Deployment (optional).** Distil into a small causal student, e.g. Cat-Nano-sized or
   Parakeet-sized, on unlabeled audio, using teacher logits.

## Pseudo-labels from ASR + LLM

TurnBench never labels EOT directly. `turnbench/gold.py` derives every EOT/INT positive and
negative mechanically from **per-segment labels** plus timing:
- Turn: Normal Turn, Strong Floor Hold, Bounded Response, Filler, Overlap.
- Floor-taking vs. non-floor-taking interruption.
- Backchannel types.
- NonContent: noise, channel bleed, non-linguistic speech.

A segment end counts as an EOT only when the floor passes to the other speaker. So the
labeler should answer the annotators' question, not "is this a turn end?":

1. **Segments:** run causal-free (offline) VAD per channel to produce VAD-segmented spans,
   matching otoSpeech's annotation style. Boundaries come from VAD or forced alignment, never
   from the LLM: the scorer tolerance is only 0.25 s before the anchor.
2. **Text:** per-channel ASR with word timestamps, e.g. Whisper-large-v3 or Parakeet-TDT.
3. **Labels:** the LLM sees an interleaved two-channel transcript with timings and future
   context. It assigns each segment a fine label from `LABEL_MAP`; future context is fine
   because these are training targets, not model inputs. Write each label as otoSpeech-style
   SRT (`[Label] text`) so `annotations()` → `gold_and_activity()` → `build_conversation_events`
   run unchanged.
4. **Segments with no transcript** (VAD fires, ASR is empty):
   - laughter or non-speech: classify with an audio tagger or the MOSS-Audio LLM;
   - channel bleed: cross-channel energy and correlation.

   These matter because NonContent spans are **INT negatives**. Laughter is currently
   ignored by the TurnBench floor rule ("Turn ∪ Laughter is deferred"), so weak laughter
   recall costs relatively little.
5. **Validate before use:** run the pipeline on otoSpeech *train* conversations and score the
   pseudo-gold against the human gold, treating the pseudo events as predictions in the
   pinned scorer. Then A/B the same audio with human vs. pseudo labels on otoSpeech dev and
   TurnBench dev. Tune labeler prompts on otoSpeech train only.

Known weak spots:
- ASR drops disfluencies and fillers, which are exactly what separates a hold from a yield.
  Prefer verbatim-style ASR.
- Overlapped speech, and floor-taking vs. non-floor interruptions, are judged at onset by
  annotators. The LLM sees the outcome, which is what the gold uses too.
- A text-only LLM ignores prosody. The model learns prosody from audio, but label noise
  concentrates on prosodically marked holds.

Candidate two-channel audio, licenses unverified unless stated:
- otoSpeech's own 104 h (human labels already exist);
- Fisher English, ~1,960 h, and Switchboard, ~260 h: LDC licenses, 8 kHz telephone, so a
  domain mismatch;
- CANDOR: check whether per-speaker channels exist.

Single-channel podcasts would need diarization and separation and are deferred. TurnBench
dev/test are a separate private recording (154 conversations, 106 speakers), so the main risk
from public corpora is domain, not leakage.

## Protocol

Inherits `training/turn_detector/PROTOCOL.md`:
- the pinned evaluator `38a6f87`;
- the frozen otoSpeech actor split (131/16/20);
- **no TurnBench test audio, labels or feedback** in training, pseudo-labeling, normalization,
  thresholds or selection;
- public TurnBench dev is development data, not an untouched holdout.

Pseudo-labeled corpora must also exclude every TurnBench dev/test file by construction.

## Files

- `cat_encoder.py`: fetches the pinned code and encoder-only weights (HTTP range reads; the
  decoder and quantizer are never downloaded) and provides `CatEncoder` (full and streaming).
  Run `python cat_encoder.py <dir>`.
- `model.py`: VAP labels, `TurnModel` and the losses.
- `test_cat_encoder.py`, `test_model.py`: contracts. `CAT_DIR=<dir>` enables the real-weight
  tests.

Next: the otoSpeech feature-caching script (stage 0) and its trainer, then a pseudo-labeler
prototype validated on otoSpeech train.
