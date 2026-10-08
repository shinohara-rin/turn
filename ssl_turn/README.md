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
- Heads (stereo), all speaker-equivariant (swapping channels swaps `HELD_0`/`HELD_1`; tested):
  - **floor now**: one joint state, `HELD_0`, `HELD_1`, `OPEN` or `CONTESTED`;
  - **floor projection**: the floor state 0.4, 0.8 and 1.6 s ahead;
  - **per-speaker act**: `SILENT`, `CLAIM`, `BACKCHANNEL`, `LAUGHTER` or `NONCONTENT`;
  - joint **256-class VAP** (bins 0.24/0.40/0.56/0.80 s over a 2 s horizon), which needs only
    activity and so works on unlabeled stereo.
- **Mono mode** for mixed-speaker audio: the same trunk without cross-attention, a mono
  embedding, and speakers in **arrival-order slots**. It has the same floor and act heads
  over slots, plus a diarization logit per slot (overlap allowed).
- Decisions every 80 ms rather than attempt 1's 160 ms grid, which removes up to 80 ms of
  quantization latency.

### Targets: floor ownership, not end-of-turn flags

The primary target is **who holds the floor** (`labels.floor_targets`):

| Floor | Meaning |
|---|---|
| `HELD_c` | c holds the floor: talking, or pausing without yielding (a hold) |
| `OPEN` | the holder yielded and nobody has claimed it yet |
| `CONTESTED` | both claim it at once |

**Overlap is a state of the floor** (contested), not a coincidence of two independent
p(speaking) values. **Vocal acts are separate from the floor:**
- a backchannel is speech that leaves the floor where it was;
- a claim is speech that bids for it; turns and interruptions are claims whether or not
  they succeed.

How contests and pauses resolve is learned through **floor projection**, the floor state at
t + 0.4/0.8/1.6 s: a floor-level VAP. TurnBench events are transitions of the floor:

| Event | Floor transition |
|---|---|
| EOT for c | `HELD_c` → `OPEN` / `HELD_other`, while c is quiet |
| Floor-taking INT for c | `HELD_other` → `CONTESTED` → `HELD_c` |
| Failed attempt | `CONTESTED` → back to `HELD_other` |
| Backchannel | never leaves `HELD_other` |

Scores for the official sweep (`labels.turnbench_scores`):
- **EOT_c** = p(`OPEN`) + p(`HELD_other`) now, committed while c is quiet;
- **INT_c** = p(`HELD_c` at +0.8 s).

How labels are derived from TurnBench gold:
- Claims and acts come from the annotated segments.
- Whether a holder's silence is a hold or a yield comes from the gold's own EOT anchors,
  the same floor-passing rule `gold.py` uses.
- No-majority spans get zero weight: turn-view disputes zero the floor target, label-view
  disputes zero the acts. Agreed failed attempts stay supervised, because they define a
  lost contest, even though the scorer excludes them.

`test_labels` checks all of this against the pinned `build_conversation_events`:
- every EOT anchor releases the floor;
- every hold pause keeps it;
- the floor-taking interruption shows as `CONTESTED`, and the projection gives it to the
  interrupter;
- the failed attempt also shows as `CONTESTED`, but the projection returns it to the holder;
- a backchannel never contests.

Beyond TurnBench, the floor is the quantity a dialogue system acts on: whether it may
speak, whether it is being talked over, whether to yield.

### Diarization objective (mono)

Mono targets put speakers in slots by order of first speech (`labels.to_slots`),
following Sortformer's arrival-time ordering ([Streaming Sortformer](https://arxiv.org/abs/2507.18446)):
- Order of first speech is decided causally, so no permutation search is needed.
- Each slot is supervised with:
  - **activity**: multi-label, so overlap is a first-class target;
  - the **floor and acts**: as in stereo, with `HELD_c` meaning slot c.

Why an explicit diarization head:
- **For the representation:** telling speakers apart and detecting overlap forces
  speaker-discriminative, overlap-aware features. These are the cues that separate a
  floor-taking interruption from a backchannel. Mono and stereo share the trunk, so stereo
  benefits too.
- **For users:** mono output says *who* holds the floor, who yielded and who barged in.
- **For data:** diarization is the cheapest podcast label. pyannote output gives slot
  activity for every podcast hour (`slot_activity_from_segments`), while floor labels need
  the ASR+LLM pass. Stereo mixed down to mono gives exact slot labels.

Open issue: **speaker memory beyond the attention window.** A speaker silent for longer
than the window (20 s by default) can only be re-identified from what is still in context.
Planned fix: an arrival-ordered speaker cache, i.e. the highest-confidence past frame
embeddings per slot kept as extra keys, as in Streaming Sortformer's AOSC. Measure
slot-swap rate versus silence length on otoSpeech dev mixed down before building it.

Evaluation:
- diarization error on otoSpeech dev mixed down, both causal and arrival-ordered;
- [`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
  as an external streaming baseline.

## Multilingual and code-switching

TurnBench is English-only, but nothing in the floor objective is. The parts that are
language-dependent:
- **Encoder coverage is unknown:** Cat's language mix is undisclosed (3M h "diverse
  audio"), and its card reports only English and Chinese. Before relying on it, run a
  per-language probe: train frozen-feature VAP and diarization on a few hours per language,
  then compare loss and slot-swap rate across languages against English.
- **Turn-taking norms differ across languages:** DuplexChat English has 3.1 backchannels/min
  and 10% simultaneous speech; Japanese has 5.6/min and 21%. So the model must not hard-code
  English floor dynamics.
  - It gets **no language-ID input**: language ID is ill-defined under code-switching.
  - An auxiliary per-frame language-ID head (soft, multi-label) is an option: it is useful
    to users and makes switches explicit, but it is not a floor input.
- **Data:**
  - DuplexChat-Ja: 132k h, same pipeline.
  - DuplexChat-Pipe filters PodcastIndex feeds by language tag, so it can be run for other
    locales; the tag is unreliable, so confirm with audio language ID.
  - Code-switching: mine bilingual-community feeds (e.g. Hinglish, Taglish, Singapore or
    Malaysian zh/en, Spanglish) by detecting switches within clips with frame-level
    language ID, rather than trusting tags.
- **Labels:**
  - Diarization and VAP are language-independent.
  - The ASR+LLM floor labeler needs multilingual, verbatim-style ASR. Code-switched ASR is
    the weak link, so prefer an ASR that tolerates switching and an LLM prompt that sees
    both languages.
  - Calibrate on a small human-labeled set per language before trusting it.
- **Evaluation:** there is no multilingual TurnBench, so build small evaluation sets
  annotated with the TurnBench protocol, 1–2 h per target language plus a code-switching
  set. LLM labels cannot be the yardstick. Existing two-speaker multilingual corpora can
  seed this and serve as fine-tuning data; channel layout and license must be checked
  first. For example, [MLC-SLM](https://arxiv.org/abs/2509.13785) has ~1.6k h of
  two-speaker, ~20-minute conversations in 11 languages with speaker labels, and its 2026
  edition adds Tagalog, Urdu and Turkish.

Order of work: get English and TurnBench right first, while keeping every target and
pipeline language-agnostic. Then add Japanese via DuplexChat-Ja, which needs no new pipeline,
as the first transfer test.

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

The main audio source is podcasts (next section). Fisher and Switchboard (LDC, 8 kHz
telephone) are a fallback. TurnBench dev/test are a separate private recording
(154 conversations, 106 speakers), so the main risk from public corpora is domain, not leakage.

## Podcasts: single-channel audio made into stereo

**DuplexChat** has already done the diarization step at scale.
- [sarulab-speech/DuplexChat](https://huggingface.co/datasets/sarulab-speech/DuplexChat)
  @`ff5c418`, arXiv 2607.04941, MIT code and manifests, no audio.
- 15.3M English clips (282k h) from PodcastIndex feeds; median clip 40 s.
- Each clip is a span where pyannote community-1 found exactly two speakers, at least 10 s
  long, with neither speaker above 80% of the talk.
- Their `reconstruct_dataset.py` downloads each episode, slices the span and separates it
  with **DialogueSidon** into L/R stereo, in resumable GPU shards.

`podcast_subset.py` picks nested subsets for scaling runs (1k → 10k h). It splits by
**feed**, because hosts recur across episodes, and caps hours per feed and per episode. On
the manifest's first 58k rows, 100 h of training data spans 222 feeds.

**Two ways to make stereo, both unvalidated:**

| | DialogueSidon (upstream) | `gated_stereo` (ours) |
|---|---|---|
| Overlap | Separated | Both talkers in both channels (like strong bleed) |
| Acoustics | **Re-synthesized**: w2v-BERT → diffusion → DAC vocoder, denoised | Original audio; the inactive channel is attenuated to a jittered bleed level |
| Risk | Vocoder domain shift; restoration may drop breaths, laughter and room tone, which are turn-taking cues present in TurnBench | INT supervision is weak where overlaps are unseparated; diarization errors go straight into the channels |
| License | **CC-BY-NC-4.0** weights | No extra model |

Gating doesn't leak future labels. The model is causal, so it sees a gain change only after
diarization says speech started or stopped, much as it would from a real channel's energy.

**Calibration before scaling (the first podcast experiment).** otoSpeech has true channels:
1. Mix otoSpeech train to mono.
2. Run pyannote, then each route.
3. Measure activity agreement with `activity_agreement`, especially overlap recall.
4. Train stage 0 three ways (real / gated / Sidon otoSpeech) and compare on TurnBench dev.

The gap between real and pseudo stereo, measured on identical conversations and labels,
says which route to use and how much podcast hours must compensate. Only then reconstruct
a 1k h subset.

## Interruption supply

TurnBench dev gold (public `dev-gold.json`, 7.31 h, 38 conversations), per hour:

| | EOT + | EOT − | INT + (floor-taking) | INT − (backchannel / noise) |
|---|---|---|---|---|
| All | 260.6 | 145.5 | 47.5 | 510.8 |

INT rate by conversation type:

| Conversation type | INT / h |
|---|---|
| Argumentative/Deliberative | 70.8 |
| Task-Oriented/Transactional | 69.5 |
| Collaborative/Problem-Solving | 62.3 |
| Instructional | 45.8 |
| Casual/Spontaneous | 19.7 |
| Narrative/Storytelling | 11.5 |

Podcasts are mostly interview, casual and narrative talk, i.e. the *low*-INT genres, and
separation is least reliable during overlap, which is where interruptions live. So
**podcasts are expected to help VAP and EOT more than INT**. Interruption data needs its own
plan:
1. **Measure** INT/h on a reconstructed sample with the pseudo-labeler before scaling.
   DuplexChat's own statistics (10% simultaneous speech, 48% of English transitions
   overlapping) count overlap, not floor-taking interruptions.
2. **Mine and oversample:** select feeds by PodcastIndex category (debate, politics, sports
   talk, panel) and clips by measured overlap rate. 282k h leaves plenty of room to be
   selective.
3. **Use mono where separation fails:** the mono route labels INT from diarization and the
   LLM without separating overlap.
4. **Keep human labels decisive for INT:** otoSpeech gets a higher INT-state weight, and the
   final fine-tune is on human labels only.

LLM+TTS synthetic dialogues sounded unnatural in a past attempt and are postponed.

## Mixed-source training

Sources differ in what they label and how much to trust it:

| Source | Input | VAP | Label states |
|---|---|---|---|
| otoSpeech train (~33 h) | real stereo | human activity | human |
| otoSpeech mixed down | mono | – | exact per slot (`slot_targets`) |
| Podcasts, separated or gated | pseudo stereo | VAD on channels | ASR+LLM pseudo |
| Podcasts, original | mono | – | slot activity from pyannote; states from ASR+LLM |

The objectives really are heterogeneous: in stereo, a channel identifies the speaker, while
in mono the model must also diarize. Both modes share the encoder and trunk, and each has
its own heads. In podcasts, agreement between pyannote and VAD on DialogueSidon's separated
channels gives a free confidence weight for the diarization targets.

How the code handles this:
- **Partial labels:** every target has a per-frame weight; a missing target is omitted or
  gets weight 0. Each loss term is normalized by its own weight mass (`model.loss`).
- **Source conditioning:** a learned source embedding (`REAL_STEREO`, `GATED_PODCAST`,
  `SEPARATED_PODCAST`) lets the model absorb pipeline artifacts instead of baking them
  into its notion of a turn. Inference always uses `REAL_STEREO`. Ablate with and without it.
- **Schedule:** pretrain on podcasts, mixing in otoSpeech at a fixed share (~10–20% of
  batches), then fine-tune on otoSpeech alone.
- **Trust:** pseudo label-state weights are scaled down (start at 0.3), or by labeler
  confidence.
- **Mono share:** mono batches are a fixed share of training. TurnBench (stereo) stays the
  selection metric; mono is reported on otoSpeech dev mixed down.

Evaluation ladder (TurnBench dev; each rung must beat the previous):
1. otoSpeech only.
2. + podcast pretraining with VAP only.
3. + podcast pseudo EOT/INT.
4. Scale hours 1k → 10k.

The rung-4 curve decides whether more GPU time for reconstruction is worth it. Dozens of
extra conversations rarely move the result, so meaningful steps here are thousands of hours.

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
- `podcast_subset.py`: feed-disjoint, nested DuplexChat subsets in the upstream schema.
- `pseudo_stereo.py`: `gated_stereo` and calibration metrics against real stereo.
- `labels.py`: TurnBench gold → floor, floor projection and acts; arrival-order slots for mono;
  score readout.
- `test_cat_encoder.py`, `test_model.py`, `test_podcast.py`, `test_labels.py`: contracts.
  `CAT_DIR=<dir>` enables the real-weight tests; `test_labels` needs the pinned `turnbench`
  importable. Install TurnBench in its own environment: its `huggingface-hub==1.17.0` pin
  breaks recent transformers.

Next:
1. otoSpeech feature caching and the stage-0 trainer.
2. Podcast-route calibration on otoSpeech mono mixes.
3. A pseudo-labeler prototype validated on otoSpeech train.

The otoSpeech split stays frozen at 131/16/20, matching attempt 1.
