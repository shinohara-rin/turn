# How fast and how sure can a human be? A real-time bound for turn-end detection

Question (Rin, 2026-10-11): how certain and how fast can a human listener decide, in real time and
without hindsight, that a turn has ended?

## Short answer

The literature has no clean number for a third-party listener scored the way our benchmarks score
models (hit rate on true ends, false-alarm rate on mid-turn pauses, latency). What it does give:

| What | Number | Source |
| --- | --- | --- |
| Fastest reaction to silence, answer already prepared | mean 283 ms after speech offset; under 200 ms only 3% of the time (215 ms to a tone onset) | Torreira & Bögels 2022, J. Phonetics |
| Timing on clips that are all real turn ends | button presses mean −186 ms (before the end), mode at 0 ms; long turns get a second peak at earlier completion points | de Ruiter, Mitterer & Enfield 2006, Language 82(3) (108 Dutch turns, 60 listeners) |
| Hold vs. shift after 500 ms of silence, 10 s of context | crowd majority vote F = 0.709; an LSTM gets 0.762; always-hold 0.421 | Skantze 2017, SIGDIAL (Map Task, 207 pauses, 3 judges each) |
| Fooled by a complete-sounding hold | ~30% of presses land at a turn-medial point with syntactic completion plus phrase-final prosody; almost none without the prosodic boundary | Bögels & Torreira 2015, J. Phonetics 52 (as summarised by later work; not checked against the full text) |
| How often listeners hear a possible end mid-turn | 16% of word boundaries in multi-unit turns are marked as response points by real-time listeners (55 turns, 118 listeners) | Umair, Sarathy & de Ruiter 2024 (EMNLP Findings); Umair & de Ruiter 2026 |
| What participants actually do | gap mode ~200 ms; ~40% of transitions in overlap (Map Task, Spoken Dutch Corpus); within-speaker pauses are longer than between-speaker gaps (medians 580-640 vs 470-520 ms); a 500 ms silence rule catches ~half of the gaps and ~57-60% of the pauses | Heldner & Edlund 2010, J. Phonetics 38 |
| Same, on TurnBench itself | transfer offset median −151 ms (smooth) / −281 ms (with interruptions); 64% of transfers start in overlap; non-overlap gap median 0.38 s; within-speaker pause median 0.51 s | TurnBench paper (arXiv 2608.25218) §IV |

Reading these together:

1. **Speed.** A human who already knows the turn is over needs ~280 ms to start speaking after the
   silence begins. Anything faster is prediction from cues before the silence (de Ruiter's −186 ms,
   the 200 ms conversational gap mode). So "~0 ms" is reachable only on the clearest ends, and only
   by committing before the speaker stops.
2. **Certainty.** Humans are not near-perfect at telling a hold from an end. After 500 ms of
   silence, third-party judges reach F ≈ 0.71 on Map Task, below a 2017 LSTM. A pause after a
   syntactically and prosodically complete clause fools listeners about a third of the time.
   Listeners perceive many mid-turn points as places they could come in, and whether the speaker
   adds more there is the speaker's choice, so part of the false-alarm rate is irreducible for any
   listener, human or model.
3. **Conversational timing is not a listener bound.** The fast gaps and overlaps in real
   conversation come from participants who co-produce the ending: the speaker often stops because
   the listener started, and early starts are repaired in overlap instead of counted as errors. A
   replayed benchmark removes that feedback, so the TurnBench paper's "humans transfer at −151 ms"
   is not a target a replay-scored system can be held to (and under TurnBench's scorer a start more
   than 250 ms before the end is a miss).

## Mapping onto our numbers

- **TurnBench EOT** (our best dev 0.975 recall / 0.069 FP / p50 534 ms, PR #12 policy). Gold is
  hindsight: an EOT positive exists because the floor actually changed hands, and annotators heard
  what came next. Annotator agreement (Cohen's κ 0.77-0.80) therefore bounds label noise, not
  real-time decisions. There is no published real-time human number on TurnBench audio. Our p50 of
  534 ms is ~250 ms slower than the ~283 ms reactive floor; our FP 0.069 at 0.975 recall is already
  far better than the ~30% fooled-by-a-complete-clause rate and the F 0.71 crowd result suggest a
  third-party human would get on mid-turn pauses (inferred; different corpora and tasks).
- **LiveKit eot-bench en** (our teacher 612 ms mean latency at 5% cut-off; LiveKit v1 543 ms;
  silence-only 1600 ms). An oracle that knew every end at silence onset would score 200 ms (the
  harness's minimum action delay) at ~0% cut-off; a human who reacts only to silence they know is
  final lands near 300 ms. Where a real listener sits between that and our 612 ms is exactly what
  the literature cannot say, so the listening test below measures it directly.

## The listening test

`turn-listening-test.html` is a browser page (published as a private artifact for Rin) that plays
all 400 eot-bench en user turns in four shuffled blocks of 100 (~20 min each). The listener is the
voice agent: they read the agent's last message, hear the caller, and press Space (or tap) when they
would start replying. The caller keeps talking after a press in a mid-turn pause, so every pause gets
its own decision as in the harness; a press after the real end stops the clip. A one-minute
reaction baseline (press when a murmur stops) separates prediction from reaction.

Scoring (`score.py`, same rules in the page):

- A press commits a reply that lands at least 0.2 s into a pause (the harness's shortest action
  delay), in the first pause still open then. A press while the caller is talking means "reply at
  the next pause".
- Cut-off: a reply inside a 0.2-5 s hold, or a hold longer than the 3.5 s timeout (harness rule).
- Latency: from the start of the final silence to the first reply there, capped at the timeout;
  no reply counts as 3.5 s.
- `score.py` also sweeps a silence timeout added to the human (human OR timeout), giving a
  cut-off/latency curve to set against model curves.
- Checks: a synthetic "reply after 1.0 s of any silence" listener scores 9.6% cut-off at 1000 ms,
  matching the harness's own VAD baseline (1000 ms at 10%); an oracle at +0.3 s scores 0.1% at
  300 ms (one hold is longer than the timeout).

Timing: presses are mapped to what was audible using the Web Audio output timestamp, so output
latency is compensated; Bluetooth headphones add delay the browser does not report, so the page asks
for wired audio. A key press also carries ~100-200 ms of motor time that a model does not pay, so
the human latency is a slightly pessimistic estimate of the human decision.

Rebuild the clips: `python build_clips.py en.parquet OUT` (eot-bench en parquet from
`livekit/eot-bench-data`, CC BY 4.0). Score: export the artifact's `trials` collection (or the page's
saved file) and run `python score.py 'trials/*.json'`.

## Backchannels

Two different questions, with different answers.

**Placing a backchannel** (when to say "mm-hm") has no single right answer, so the ceiling is low.
Listeners differ in where they backchannel: Ward & Tsukahara's low-pitch rule predicts Japanese
backchannels at 56% coverage / 34% accuracy (English 48% / 18%), about half of its false alarms are
places where this listener stayed quiet but another might not, and the authors estimate that a human
eavesdropper predicting the actual listener would reach only ~95% coverage at ~67% accuracy
(Ward 1998, summarising Ward & Tsukahara; an estimate, not a measurement). Real-time listeners asked to
backchannel wherever they could mark many points per turn and disagree on most of them
(Umair et al. 2024). Any backchannel model scored against one listener's choices tops out around
two-thirds precision.

**Telling a backchannel from a takeover** (TurnBench INT, where backchannels and non-content
vocalisations are the negatives) has no published real-time human number that I could find.
Useful facts:

- TurnBench dev has 3,733 INT negative spans vs 347 positives. Their durations (gold, from
  `dev-gold.json`): median 0.88 s; only 4% are over by 0.4 s, 15% by 0.5 s, 41% by 0.75 s, 56% by
  1.0 s. At our INT p50 of 434 ms almost none has finished, so neither a model nor a human can wait
  for the vocalisation to end; the call has to come from its first word or two.
- TurnBench excludes non-floor-taking interruptions from scoring because "at onset it cannot be
  distinguished from a floor-taking interruption" and the label depends on what happens seconds later.
- The person being interrupted is the in-situ human detector, and they keep talking a median 1.48 s
  after a floor-taking interruption starts before ceding (TurnBench paper §IV). Part of that is
  finishing a phrase on purpose, so it is an upper limit on human detection time, not a measure of it.
- Our own INT fires on backchannels are mostly multi-word ones (fire rate 4% for one-word
  backchannels, 48% for four or more words on TB dev; see the semantics thread), which are also the
  ones a human would find ambiguous at onset.

A real-time human number would need a TurnBench block in the test: play the conversation up to a
listener vocalisation onset, then let the listener press "taking the floor" as soon as they can
tell. TurnBench audio is gated and was not reachable from this session.

## Sources

- de Ruiter, Mitterer & Enfield (2006). Projecting the end of a speaker's turn. Language 82(3):515-535.
- Bögels & Torreira (2015). Listeners use intonational phrase boundaries to project turn ends. J. Phonetics 52:46-57.
- Torreira & Bögels (2022). Vocal reaction times to speech offsets. J. Phonetics.
- Heldner & Edlund (2010). Pauses, gaps and overlaps in conversations. J. Phonetics 38(4):555-568.
- Skantze (2017). Towards a general, continuous model of turn-taking. SIGDIAL. https://aclanthology.org/W17-5527
- Levinson & Torreira (2015). Timing in turn-taking and its implications for processing models of language. Front. Psychol. 6:731 (production latency ≥600 ms for a single word, hence planning in overlap).
- Umair, Sarathy & de Ruiter (2024). Large language models know what to say but not when to speak. Findings of EMNLP. https://arxiv.org/abs/2410.16044
- Umair & de Ruiter (2026). Using semantic uncertainty to estimate transition relevance in turn-taking. https://arxiv.org/abs/2609.10934
- TurnBench (2026). arXiv 2608.25218.
- Ward (1998). Some exotic discourse markers of spoken dialog. ACL/COLING workshop on discourse relations and markers, W98-0311. https://aclanthology.org/W98-0311
