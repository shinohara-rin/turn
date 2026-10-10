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
   IndexTTS-2.5 per whole item, in dialogue order ◄─────────────────────────┘
   (rolling speaker prompt, script emotion; forced-aligned word timings)
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

## Prosody: giving the TTS the dialogue

MultiTalk renders each short utterance with IndexTTS2 in isolation, from a
fixed reference clip. The model never sees what came before or after, so a
turn split at a pause ends like a finished sentence, and each turn starts
from the same neutral register whatever the other speaker just did. The
IndexTTS backend (`--tts indextts`, IndexTTS-2.5 by default, IndexTTS2 with
`--index-version 2`) gives it the context it can take:

| what | how | why it matters here |
|---|---|---|
| whole turns | one call per item; `<pause X>` becomes a comma, then the pause is cut back in at the forced-aligned word boundary at its sampled length; IndexTTS's 120-token segment split is turned off | a hold pause keeps continuation intonation instead of turn-final falling pitch, which is exactly the EOT hard negative TurnBench scores |
| rolling speaker prompt | the prompt is the speaker's bank clip (timbre anchor, 5 s) followed by their most recent rendered speech, up to 14 s; items are synthesized in script order | rate, energy and register carry from turn to turn, and IndexTTS's mel stage continues from the end of the prompt, i.e. from what this speaker just said |
| script emotion | optional per-item `"emotion": {"surprised": 0.5}` written by the script LLM, which sees the whole dialogue; mapped onto IndexTTS2's 8 emotion axes | delivery follows what was just said (a sharp retort, a surprised "oh wow") |
| entrainment (opt-in) | `--entrain 0.3`: lines without a script emotion use the partner's last line as emotion reference at that strength | listeners match the energy of who they answer |

What it cannot do: IndexTTS conditions on audio and an emotion vector, not
on the other speaker's words, so cross-speaker coherence comes from the
prompt and the script only. Dialogue-native TTS (MOSS-TTSD, FireRedTTS-2,
VibeVoice) model both speakers jointly and would be the next step if that is
still the weak spot; they would need per-speaker stems and word timings to
fit this pipeline.

On the casual example (IndexTTS-2.5 on CPU, GLOBE voices, whisper `base.en`),
context vs the MultiTalk-style ablation (`--no-dialogue-context`: chunk by
chunk from the fixed bank clip):

| | with context | without |
|---|---|---|
| pitch at the end of a scripted hold pause (re speaker median / slope over the last 0.5 s) | +2.1 st / rising +0.8 | -1.8 st / falling -1.0 |
| pitch at a turn end (statements) | -1.5 st / falling -0.9 | -2.9 st / falling -1.6 |
| speaking rate, median over turns | 3.3 words/s | 3.2 words/s |
| ASR WER | 9.0% | 9.1% |

Without context a hold pause falls like a turn end; with it the voice stays
up and the turn end still falls, which is the cue an EOT model should learn.
Only 7 hold pauses and 16 turn ends, so read it as direction, not size.
The first version fed the speaker's own pauses back into the prompt and
slowed every speaker turn by turn (3.6 down to about 2 words/s). The prompt
is now the 8 s bank clip plus at most the last 3 s of the speaker's own
speech, with silences squeezed to 0.15 s; `duration_factor` is only set
from a script line's `speed`.

### Output quality (audit, 2026-10-09)

IndexTTS copies the recording quality of its prompt, and anything fed back
into the prompt compounds line by line. Our calls match the reference
`infer` parameters and text handling; the quality loss came from three of
our own choices, measured on the same casual lines with torchaudio SQUIM
(reference-free PESQ / SI-SDR) and whisper WER:

| setup | PESQ | SI-SDR | WER | words/s |
|---|---|---|---|---|
| reference usage, GLOBE prompt | 2.93 | 14.7 dB | | |
| reference usage, LibriTTS-R prompt | 3.49 | 21.4 dB | | |
| old per-line rolling prompt with pace controller | 2.54 | | | |
| clean 8 s prompt + last 3 s of history | 3.51 | 22.4 dB | | |
| 120-190 word pass, GLOBE prompt | | | 33% | 5.6 |
| up to 60 word pass, clean prompt | 3.52 | | 6.5% | 3.14 |

1. GLOBE (Common Voice) prompts carry phone-mic noise into every line;
   the bank now defaults to LibriTTS-R.
2. The pace controller pushed `duration_factor` toward 0.7 and the
   compressed speech then went into the next prompt; it is removed.
3. Passes over ~60 words rush and garble; `floor`/`speaker` passes are
   capped at 60 words.

Line ends: lines used to be cut 20 ms after the forced-aligned end of the
last word, which lands before the voice and breath have died away, so most
lines stopped dead (casual: median level in a line's last 10 ms 15 dB under
its peak, 90% of lines above -30 dB). `_trim` and `_retime` now cut where
the level has stayed 45 dB under the peak for 80 ms (so a final stop's
release is kept), at most 0.5 s past the aligned end, with a 50 ms fade:
median end level -63 dB, 3% of lines above -30 dB, median tail 140 ms.
Word times still mark the words, so placement and labels are unchanged.

Full casual render after the quality fixes vs the earlier `floor` render: PESQ
3.46 vs 3.12, SI-SDR 20.0 vs 19.2 dB, 3.1 vs 3.2 words/s, WER 7.2% vs 9.8%
(argumentative: WER 10.9%).

### Reading several lines in one pass

Per-line calls still start every line from scratch. `--index-pass` (default `floor`) lets
IndexTTS read several of a speaker's lines in one call; render aligns the
pass and cuts it back into lines at the silences between them (a line the
model garbled is re-synthesized alone).

| casual example | `turn` (one call per line) | `floor` | `speaker` |
|---|---|---|---|
| what one call reads | one line | a speaker's lines until the other takes the floor | up to 60 words of a speaker's lines |
| turn end: pitch level / slope | -1.5 st / -0.9 | -2.3 st / -1.4 | -0.9 st / +0.2 |
| scripted hold pause: level / slope | +2.1 st / +0.8 | +1.2 st / +0.2 | +0.8 st / +0.4 |
| pitch jump into the speaker's next line | 1.2 st | 1.1 st | 1.6 st |
| speaking rate (median) | 3.3 words/s | 3.3 | 3.6 |
| ASR WER | 9.0% | 9.8% | 8.9% |

`speaker` turns real turn ends into mid-reading sentence ends, so they stop
falling, which removes the cue an EOT model needs; `floor` keeps it while
joining a speaker's lines within one floor. Per-line emotion vectors are
ignored inside a multi-line pass.

Voice prompts come from LibriTTS-R (CC BY 4.0, restored studio-quality
audiobook speech; gender from median F0) by default, or GLOBE_V2 (CC0 Common
Voice, more accents but noisy) with `--source globe`: `turnsynth voices`
joins a few utterances per speaker into a 6-10 s clip. Nothing from TurnBench is used as a voice.

## Japanese and Chinese

`turnsynth scripts --language ja|zh` writes Japanese or Mandarin scripts;
IndexTTS-2.5 reads both natively (`lang="ja"|"zh"`). What changes per
language (`turnsynth/lang.py`):

- **Words.** `after_word` and the turn-length rules count words, so the
  LLM writes ja/zh with a space between words ("昨日 さ、 駅 前 の カフェ に
  行っ た"); the filter rejects unsegmented scripts (`unsegmented`). The
  spaces are removed before the TTS reads the text.
- **Prompting.** Both passes get the language's rules: setting and names,
  register, its punctuation (、。？ / ，。？), fillers, and its backchannel
  inventory and placement (Japanese aizuchi at phrase boundaries inside the
  other's turn; a turn ending in けど/から/し stays open). The backchannel
  target is scaled from TurnBench's English rates by 1.6 for Japanese and
  0.8 for Mandarin, following the ordering in Clancy et al. (1996) on
  reactive tokens in English, Japanese and Mandarin; the factors are rough
  and the timing model is otherwise TurnBench's.
- **Alignment.** MMS_FA aligns a romanization of each word (pykakasi for
  Japanese, read in sentence context; pypinyin for Chinese):
  `pip install "synth[cjk]"`.
- **Annotation.** ASR needs a multilingual Whisper (`--asr small`); the
  error rate is per character for ja/zh; the rule judge and the LLM judge
  know the language's backchannels; parquet metadata carries `language`.
- **Voices.** Bank entries carry a `language`, and a script only draws
  voices of its own language. `--voice-bank` takes several banks,
  comma-separated:

```bash
hf download AISHELL/AISHELL-3 --repo-type dataset --include "test/wav/*" spk-info.txt --local-dir aishell3
turnsynth voices aishell3 --source aishell3 --out voices-zh --per-gender 20     # studio Mandarin, Apache 2.0
hf download TTS-AGI/emilia-yodas --repo-type dataset --include "JA/JA-B000000.tar" --local-dir emilia
turnsynth voices emilia/JA/JA-B000000.tar --source emilia --language ja --out voices-ja --per-gender 20  # CC BY 4.0
turnsynth render scripts-ja --tts indextts --voice-bank voices-ja,voices-zh,voices --asr small ...
```

Examples: `examples/scripts/casual_kyoto_ja.json`, `casual_chengdu_zh.json`
(hand-written in the pass-2 format, like the English ones).
Rendered on a 3090 with `--asr small`: both accepted, agreement 0.98;
CER 12% (ja, on the kana reading) and 13% (zh). Prompt banks (SQUIM, per
clip): AISHELL-3 PESQ 3.41 / SI-SDR 21.9 dB, Emilia-YODAS JA 3.38 / 21.2 dB,
against LibriTTS-R 4.09 / 26.7 dB, so the ja/zh voices are a little less
clean than the English ones.

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

### IndexTTS (separate env)

IndexTTS pins torch 2.8 and transformers 4.52, which clash with Kokoro and
`turnbench`, so it gets its own environment; score from the main one.

```bash
git clone https://github.com/index-tts/index-tts && git -C index-tts checkout d9e41aac
uv venv -p 3.11 itts && source itts/bin/activate
uv pip install --no-sources -e ./index-tts "torch==2.8.*" "torchaudio==2.8.*" faster-whisper
uv pip install -e "synth[llm,asr]"
hf download IndexTeam/IndexTTS-2.5 --local-dir ckpt/IndexTTS-2.5
hf download mythicinfinity/libritts_r --repo-type dataset --include "data/dev.clean/*.parquet" --local-dir libritts
turnsynth voices libritts/data/dev.clean/*.parquet --out voices --per-gender 20
turnsynth render synth/examples/scripts --tts indextts --index-model-dir ckpt/IndexTTS-2.5 --voice-bank voices \
    --asr base.en --wav --out out/indextts
```

## Generating at scale

```bash
export ANTHROPIC_API_KEY=...
turnsynth scripts --n 600 --minutes 8 --out out/scripts          # round-robins the six types
modal run modal_render.py --scripts out/scripts --out out/synth --judge llm --asr small.en
turnsynth stats out/synth/parquet
```

`modal_render.py` fans rendering out over L4 GPUs with IndexTTS-2.5
(`TURNSYNTH_TTS=kokoro` for the Kokoro image); it needs a Modal secret
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
| `turnsynth/tts.py` | IndexTTS (contextual), Kokoro and dummy backends with word timings |
| `turnsynth/align.py` | MMS_FA forced alignment of script words (IndexTTS reports no timings) |
| `turnsynth/voicebank.py` | voice-prompt banks: LibriTTS-R, GLOBE_V2 (en), Emilia-YODAS (ja), AISHELL-3 (zh) |
| `turnsynth/lang.py` | per-language words, punctuation, error-rate units, romanization, backchannel lists |
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
- **Prosody with Kokoro.** Kokoro has no emotion control and is still
  called chunk by chunk, so with `--tts kokoro` a turn split at a pause ends
  each chunk like a sentence. Use `--tts indextts` for anything you listen to.
- **No laughter, breaths, noise or channel bleed labels.** `--bleed-db` mixes
  bleed in but nothing labels it as `Channel Bleed`.

TurnBench data is non-commercial and prohibits voice cloning; this pipeline
uses none of its audio, only its published statistics and scorer.
