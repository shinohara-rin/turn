# turnsynth: synthetic TurnBench

Synthetic two-channel conversations in exactly the format of
[TurnBench](https://github.com/SesameAILabs/turnbench) (arXiv 2608.25218), so
every TurnBench tool (gold construction, `turnbench.score`, `turnbench.sweep`,
the baselines' `--dataset` flag) runs on them unchanged. The pipeline follows
the MultiTalk data engine (arXiv 2609.36903) and adds what MultiTalk lacks:
event-level turn-taking labels from two independent sources.

```
 seeds + type ──► LLM pass 1: scenario, 2 personas
                  LLM pass 2: script with labelled turn-taking items ──► 4-layer filter (reason codes)
                                                                              │
             Kokoro TTS per item, word timings ◄──────────────────────────────┘
                  │
     timeline sampled from TurnBench's own timing stats (FTO, pauses, yield)
                  │
          2-channel audio ──► VAD segments ──┬─► a: generator intent  (script label of the item)
                                             ├─► b: ASR + LLM judge   (never sees the script)
                                             └─► c: overlap geometry  (sees only the VAD tracks)
                                                        │
                       TurnBench parquet (3 "annotator" tracks per speaker) + intent JSON
                                                        │
                       turnbench.gold: 2-of-3 consensus, excluded regions, EOT/INT events
```

## How it maps onto TurnBench

TurnBench segments audio with VAD, has three people label every segment with
one of 17 fine labels, and builds gold from 2-of-3 consensus at scoring time
(`turnbench/gold.py`). Here the three label tracks come from three
independent views of the same audio:

| slot | view | what it catches |
|---|---|---|
| a | **generator intent**: the label the LLM wrote for the script item the segment came from | what was meant: backchannel kind, cooperative vs competitive, bounded response, floor hold |
| b | **post-hoc judge**: faster-whisper transcribes each segment; an LLM labels every segment from the timed, interleaved transcript only | what a listener would hear: TTS that swallowed a word, a "backchannel" long enough to read as a turn |
| c | **geometry**: rules over the two VAD tracks (who was mid-speech at onset, who stopped) | what actually happened on the timeline after rendering |

Where rendering did not realize the script (an interruption that landed as
the host was finishing anyway, a host that paused right after the barge-in),
the views disagree and TurnBench's own consensus turns the segment into an
excluded interval, the same way it treats no-majority regions in the human
data. Nothing in the scorer is modified.

## Differences from MultiTalk

- **Timing is sampled, not compressed.** MultiTalk concatenates turns and
  squeezes gaps into a fixed 0.2-0.6 s overlap. TurnBench scores timing, so
  `config.py` samples floor-transfer offsets, within-turn pauses and the
  interrupted speaker's talk-on time from distributions calibrated on
  TurnBench §IV-B, per conversation type (Table III).
- **Turn-taking items are anchored on words.** Backchannels and interruptions
  name a host turn and the word they come in after; the renderer uses TTS word
  timings to place them and to cut an interrupted host on a word boundary.
- **Within-turn pauses are first-class** (`<pause X>` markers): they are
  TurnBench's EOT hard negatives.
- **Dyadic, English, six conversation types** to match TurnBench rather than
  multi-party bilingual.

## Results so far

Two hand-written scripts in `examples/scripts/` (one Casual, one
Argumentative, about 3 minutes of audio), rendered on CPU with Kokoro and
whisper `base.en`, rule judge:

| | synthetic | TurnBench (human) |
|---|---|---|
| floor-transfer offset, median | -95 to -150 ms | -151 ms (excl. interruptions) |
| transfers starting in overlap | 65-69% | 64% |
| ASR WER vs script | 4.6-5.9% | n/a |
| scripted barge-ins surviving consensus as INT positives | 3 of 4 | n/a |

The official VAP checkpoint (TurnBench's best baseline, oto fine-tune) scored
with its published test thresholds on these two conversations: EOT recall
0.865, FPR 0.000, p50 440 ms (TurnBench test: 0.845 / 0.055 / 368 ms); INT
recall 0.667 (2 of 3), p50 1625 ms. Two conversations are a smoke test, not
an evaluation; the numbers only show the data is in the right regime.

## Quickstart (CPU)

```bash
cd synth
uv venv -p 3.11 && source .venv/bin/activate
uv pip install -e ".[tts,asr,llm,score,dev]" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip uninstall spacy-curated-transformers   # pulled in by misaki, breaks with transformers 5
pytest                                         # dummy TTS, no models

# Render the examples (Kokoro + whisper, ~3 min on 4 cores) and score VAP on them.
turnsynth render examples/scripts --tts kokoro --asr base.en --device cpu --wav --out out/kokoro
turnsynth stats out/kokoro/parquet
python -m turnbench.score predictions.json --dataset out/kokoro/parquet
# any TurnBench baseline: python -m baselines.vap.predict --dataset out/kokoro/parquet --threshold-eot 0.9161 --threshold-int 0.8591
```

## Generating at scale

```bash
export ANTHROPIC_API_KEY=...
turnsynth scripts --n 600 --minutes 8 --out out/scripts          # round-robins the six types
modal run modal_render.py --scripts out/scripts --out out/synth --judge llm --asr small.en
turnsynth stats out/synth/parquet
```

`modal_render.py` fans rendering out over L4 GPUs; it needs a Modal secret
named `anthropic` (with `ANTHROPIC_API_KEY`, used by the LLM judge). Run it
from a machine where the Modal CLI can connect.

Outputs: `parquet/` (TurnBench layout, numeric ids from 900000 so they never
collide with real TurnBench ids), `intent/` (the script plus where every item
landed, word by word, for dense training targets) and `render_report.jsonl`
(agreement, WER, dropped items and reject reasons per conversation).

## Layout

| file | role |
|---|---|
| `turnsynth/generate.py` | two-pass LLM script synthesis, prompts, topic seeds |
| `turnsynth/script.py` | script schema, parser, four-layer filter with reason codes |
| `turnsynth/config.py` | TurnBench conversation types and the timing model |
| `turnsynth/tts.py` | Kokoro and dummy backends with word timings |
| `turnsynth/render.py` | timeline placement, interruption cuts, mixing |
| `turnsynth/vad.py`, `annotate.py` | VAD segments, annotators a and c |
| `turnsynth/judge.py` | ASR + LLM judge (annotator b), lexical fallback |
| `turnsynth/export.py` | TurnBench parquet writer |
| `turnsynth/stats.py` | corpus stats through `turnbench.gold` |

## Not done yet

- **No LLM has run in this pipeline yet.** The cloud environment this was
  built in has no Anthropic API key, so the examples are hand-written in the
  pass-2 format and the judge used the lexical fallback (which shares timing
  rules with annotator c, so it is far less independent than the LLM judge).
  `tests/` covers both LLM paths with a fake model.
- **Not compared against TurnBench dev yet.** Dev is gated on Hugging Face
  and the token here gets 403. With access, run `turnsynth stats` on both and
  score baselines on dev vs synthetic to see how well synthetic scores
  predict real ones.
- **Prosody.** Kokoro has no emotion or emphasis control, and each item is
  synthesized in isolation, so turn-final intonation (a main EOT cue) is
  whatever the TTS does at a sentence end. IndexTTS2 (MultiTalk's choice)
  or a dialogue-aware TTS would fit behind the same `TTS` interface.
- **No laughter, breaths, noise or channel bleed labels.** `--bleed-db` mixes
  bleed in but nothing labels it as `Channel Bleed`.

TurnBench data is non-commercial and prohibits voice cloning; this pipeline
uses none of its audio, only its published statistics and scorer.
