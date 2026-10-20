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
| script delivery | per-item `"emotion": {"surprised": 0.5}` (IndexTTS2's 8 emotion axes) and `"speed": 1.2` (rate relative to the speaker's `"pace"`, set per speaker in the design), written by the script LLM, which sees the whole dialogue; speed becomes IndexTTS-2.5's `duration_factor` | delivery follows what was just said (a sharp retort, a surprised "oh wow"), and pace varies line to line instead of being guessed from the words |
| text emotion (opt-in, not recommended) | `--text-emotion`: lines without a script emotion get one from IndexTTS's QwenEmotion text classifier | sounded exaggerated in listening, see below |
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

Delivery A/B (casual example, same voices and seed, IndexTTS-2.5 floor
passes on a Modal L4, whisper `small.en`; per speaker A / B, 12-13 lines each):

| | none (audiobook prompt sets it) | script emotion + speed | `--text-emotion` | script speed + `--text-emotion` |
|---|---|---|---|---|
| line-to-line pitch spread (SD of line median F0) | 0.9 / 0.9 st | 2.2 / 2.6 st | 2.6 / 4.3 st | 2.6 / 4.9 st |
| pitch range within a line (p90-p10) | 7.2 / 6.8 st | 9.0 / 8.5 st | 10.9 / 10.8 st | 10.6 / 10.9 st |
| speaking rate, median (A pace 1.1, B 0.92) | 3.1 / 3.3 words/s | 4.0 / 3.0 | 3.4 / 3.6 | 4.0 / 3.5 |
| rate variation across lines (CV) | 0.16 / 0.14 | 0.19 / 0.22 | 0.18 / 0.22 | 0.12 / 0.19 |
| estimated PESQ (SQUIM) | 3.43 | 3.44 | 3.55 | 3.47 |
| ASR WER | 7.4% | 7.5% | 6.1% | 10.3% |

Any explicit emotion vector moves the read away from the prompt's flat
audiobook delivery. By ear (Rin, 2026-10-10) script delivery is slightly
better than none, and `--text-emotion` is clearly worse: exaggerated,
unnatural emotion, so more pitch movement is not the goal and the flag
stays off by default.

### Catching glitches

IndexTTS sometimes puts a noise burst or a laugh into a line. Scored on
2 s windows with SQUIM's reference-free PESQ estimate, the worst window of
a line finds them (quality.py): on the casual example the two lowest
windows, 1.5 and 2.6, were the two spots heard as broken. A line whose
worst window is under `--retry-below` (2.9) is synthesized again on its
own, up to twice, and the best take kept; after that the example's worst
window was 2.7.

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
joining a speaker's lines within one floor. IndexTTS takes one emotion and
one rate per call, so a pass also ends where a line's speed differs from
the pass so far by more than 0.1 or its emotion by more than 0.3 (L1); the
pass is read with its lines' mean emotion and word-weighted mean speed.

Voice prompts come from LibriTTS-R (CC BY 4.0, restored studio-quality
audiobook speech; gender from median F0) by default, or GLOBE_V2 (CC0 Common
Voice, more accents but noisy) with `--source globe`: `turnsynth voices`
joins a few utterances per speaker into a 6-10 s clip. Nothing from TurnBench is used as a voice.

### When overlays start

A backchannel or interruption names the host words it follows by quoting
them (`"after": "it's basically a swamp."`); parse resolves the quote to a
word count. LLMs miscount words but copy text reliably, and hand-counted
anchors had put an "Oh no." before the bad news. The prompt asks that
anything an overlay reacts to come before the quote. The renderer then adds
a listener's reaction time after the anchor word's aligned end: 100-350 ms
for backchannels, which also move to the next phrase end (punctuation or a
150 ms gap) within four words, and a lognormal around 300 ms (120-800 ms)
for interruptions. At a turn change the sampled floor-transfer offset may
be negative, but the reply never starts before the other speaker's last
word (overlap there is projection of an ending the listener can hear), and
overlaps a question by at most 100 ms, since the answer needs the
question; before this, 39% of turn changes on the examples started earlier
than the last word, such as an answer starting mid-question. A floor-taking interruption's host trails off at its
own next pause after the minimum yield time, not mid-word.

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
CER 12% (ja, on the kana reading) and 13% (zh). With script emotion and
pace, glitch retries and the turn-timing fixes (Modal L4, 2026-10-11):
agreement 0.98 for both, CER 9.2% (ja) and 2.9% (zh). The earlier zh
figure was mostly Whisper writing Traditional characters and digits,
which the error rate now folds to Simplified (OpenCC) before comparing.
Prompt banks (SQUIM, per
clip): AISHELL-3 PESQ 3.41 / SI-SDR 21.9 dB, Emilia-YODAS JA 3.38 / 21.2 dB,
against LibriTTS-R 4.09 / 26.7 dB, so the ja/zh voices are a little less
clean than the English ones.

## Fish Audio S2 backend (2026-10-11)

`--tts fish` (`turnsynth/fish.py`) renders through the Fish Audio API
(`s2.1-pro-free` by default, `--fish-model`), so it needs no GPU. It plugs
into the same pipeline as IndexTTS: same voice banks (the bank clip plus its
transcript is an inline reference; clips are transcribed once with
faster-whisper into the bank's `transcripts.json`), floor passes, the
speaker's previous line as a second reference, the SQUIM glitch retry. Script
emotion becomes an S2 tag (`[happy]`, `[surprised]`, ...) on the pass when its
strongest axis is >= 0.3; S2 does not speak tags and the timestamps omit them.

**Speed.** The script's per-line speed is not passed on by default. S2's
`prosody.speed` time-stretches the output: the same sentence went from SQUIM
PESQ 4.3 (speed 1) to 2.8-3.1 (0.92 or 1.1), barely changing its length at
0.92. With speed on, the casual example scored PESQ 2.6 and the glitch retry
re-synthesized about half the lines; the emotion tags made no difference
(2.61 with, 2.62 without). `--fish-speed` turns it back on.

**Word timings.** The API's `/v1/tts/stream/with-timestamp` returns word
(en) or character-run (ja/zh) times on an 80 ms grid. `map_segments` maps them
onto the script words (character matching, so ja/zh runs are split by
character); every one of ~380 calls mapped, MMS_FA is the fallback
(`--fish-timings mms` forces it). Against MMS_FA on the same audio, Fish starts
are ~40 ms earlier and ends ~30 ms later (median). At pauses (a quiet run of
>= 150 ms, 35 dB under the line's peak), the silence's own edges show which is
closer:

| error vs the pause edge, median (mean abs) | en (65 pauses) | zh (78) | ja (157) |
|---|---|---|---|
| word end, Fish | -140 ms (144) | +5 ms (69) | +20 ms (63) |
| word end, MMS_FA | -232 ms (239) | -196 ms (212) | -163 ms (169) |
| next word start, Fish | +10 ms (45) | -25 ms (53) | -25 ms (40) |
| next word start, MMS_FA | +76 ms (86) | +18 ms (38) | +30 ms (51) |

So Fish's ends are much closer to where the sound stops (MMS_FA ends early,
which is why `_trim` cuts at acoustic decay), and starts are comparable.

**Fish vs IndexTTS.** Same scripts as the v4 IndexTTS samples, scored the
same way from the stereo audio (`--asr small.en` / `small`; error rate per
channel against that speaker's script words, so it is higher than the
pipeline's own WER; SQUIM on 2 s windows of speech). One conversation each,
so read these as a first look; SQUIM is trained on English. The ja Fish
render drew different Emilia voices than the IndexTTS one.

| | WER/CER | SQUIM PESQ median | PESQ p10 | windows < 2.9 |
|---|---|---|---|---|
| casual en, IndexTTS v4 | 8.3% | 3.61 | 3.21 | 3% |
| casual en, Fish | **3.4%** | **4.22** | **3.89** | 0% |
| casual en, Fish with speed | 4.2% | 2.61 | 1.82 | 63% |
| argumentative en, IndexTTS (natural ends) | 16.2% | 3.42 | 2.65 | 22% |
| argumentative en, Fish | **9.1%** | **4.22** | **3.91** | 0% |
| Chengdu zh, IndexTTS v4 | 8.0% | 3.59 | 2.74 | 12% |
| Chengdu zh, Fish | **4.1%** | **3.96** | **3.68** | 0% |
| Kyoto ja, IndexTTS v4 | **12.5%** | 3.17 | 2.73 | 23% |
| Kyoto ja, Fish | 15.8% | **3.86** | **3.40** | 0% |

Pipeline numbers for the Fish renders: WER 2.4% (casual), 5.9%
(argumentative), CER 2.8% (zh), 12.3% (ja); agreement 0.96-1.0. Rendering ran
at about real time (one sequential call at a time, 2-5 s per line), with no
GPU. Samples: `synth-samples/fishaudio/` in the project files.

## Grounded tapes for a full-duplex text LLM (2026-10-20)

Data for a DuplexCascade-style model (arXiv 2603.09180): a text LLM that, every 640 ms cell, reads
what the agent's ASR and VAD delivered and writes the agent's action and next few words. A script
written in one pass sees every line; the two speakers in it must not. Three additions keep a script
honest about that before it becomes training data:

- **Timing without audio.** `--tts duration` (`duration.py`) times words from syllables over an
  articulation rate, with phrase-final lengthening and punctuation pauses, so a script places and
  checks in milliseconds.
- **Causality** (`ground.py`, `render(ground=Grounding())`). A line that depends on the other speaker's
  words (`responds_to`, and for the agent its `after` anchor) starts no earlier than its speaker
  could have perceived them and reacted. People recognize a word partway in and react about 200 ms
  later; the agent gets a word only once its ASR commits it, decides only at cell boundaries and is
  heard `tts_lead` later. A barge-in stops the agent by its pipeline's rule: a hard cut 150 ms after
  the user's onset (Frankenvoice) or at the next cell after its ASR delivers the words (DuplexCascade).
  Every move is logged (`Rendered.moves`).
- **Theory of mind.** Speakers declare private `facts`; lines declare the facts they `uses` and the
  words that `reveals` one. `leaks()` reports every line that uses a fact its speaker can't know yet,
  with a reason: `never_said`, `not_yet_said`, `not_yet_heard`, `cut_off`, `spoken_over`.

`examples/ground/interstellar.json` exercises all of it: the agent's "you only have two hours" leaks a
fact the user never said, and the user's later question about "the black hole part" leaks words the
hard cut stopped before they were said (under DuplexCascade's slower stop they are said, but over the
user's own barge-in).

`tape.py` turns a rendered script into the agent's cells (`cells()`): user words as its ASR commits
them, a VAD flag, the agent's state (silent, speaking, cut off) and action. History is what was heard:
words the agent sent but never got to say are taken off the tape, and `[cut]` marks where its voice
stopped. `micro_turn()` renders a cell as DuplexCascade's control tokens in Gemma 4's turn markers.

### Zero-shot pilot

`pilot.py` runs an untrained model in that loop before anything is trained: the tape so far as the
prompt, and per cell a GBNF grammar that admits only the actions its state allows and only as many
words as its TTS queue has room for. The user is a scripted person (an opening turn, then a
backchannel or barge-in timed off the agent's audio as they hear it), and the agent's pipeline
(ASR delay, VAD, hard cut, word timings) is simulated in virtual time. `examples/pilot/scenarios.json`
has 24 probes in Full-Duplex-Bench's four categories: pause handling, turn end, user backchannel,
user interruption.

```bash
llama-server -m gemma-4-12b-it-qat-q4_0.gguf -c 4096 -np 1 --swa-full --ctx-checkpoints 0 --port 8090
python -m turnsynth.pilot examples/pilot/scenarios.json --agent llama --url http://127.0.0.1:8090 --out out/pilot
python -m turnsynth.pilot examples/pilot/scenarios.json --agent rules --out out/pilot-rules   # baseline
```

On CPU, use `-lm none` so the repacked weights aren't kept twice, and `--ctx-checkpoints 0`
(the default 32 SWA checkpoints took 5 GB).

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
| `turnsynth/fish.py` | Fish Audio S2 API backend; maps the API's word timestamps onto script words |
| `turnsynth/align.py` | MMS_FA forced alignment of script words (IndexTTS reports no timings) |
| `turnsynth/voicebank.py` | voice-prompt banks: LibriTTS-R, GLOBE_V2 (en), Emilia-YODAS (ja), AISHELL-3 (zh) |
| `turnsynth/lang.py` | per-language words, punctuation, error-rate units, romanization, backchannel lists |
| `turnsynth/render.py` | timeline placement, interruption cuts, mixing |
| `turnsynth/vad.py`, `annotate.py` | VAD segments, annotators a and c |
| `turnsynth/judge.py` | ASR + LLM judge (annotator b), lexical fallback |
| `turnsynth/export.py` | TurnBench parquet writer |
| `turnsynth/stats.py` | corpus stats through `turnbench.gold` |
| `turnsynth/duration.py` | text-only duration model as a TTS backend (`--tts duration`) |
| `turnsynth/ground.py` | causality bounds, the agent's stop rule, belief-leak check |
| `turnsynth/tape.py` | the agent's cell tape: readable lines and DuplexCascade micro-turns |
| `turnsynth/pilot.py` | closed-loop zero-shot pilot: grammar-constrained LLM agent vs a scripted user |

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
