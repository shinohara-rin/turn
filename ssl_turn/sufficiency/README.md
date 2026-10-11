# Reply sufficiency: when has the speaker said enough to answer? (2026-10-11)

Rin's question: instead of raw endpointing, predict when the speech contains enough content to wake
an LLM, possibly mid-utterance (early processing, meaningful interruptions). This folder defines
that target, labels it with an LLM in hindsight, and checks whether anything causal predicts it.

## Definition and labels (`suff_labels.py`)

A point in a turn is **sufficient** when the reply drafted from the words so far is as good a
response to the whole turn as a reply drafted after hearing everything.
- Candidates: every segment end inside a turn (`pause`) and clause punctuation inside segments
  (`mid`, time interpolated by character position), up to 12 per turn, plus the turn end.
- Call 1 drafts a short listener reply from (other speaker's last turn, words so far). One batch
  never holds two prefixes of the same turn; otherwise the model reads later words in a neighbour.
- Call 2 rates 0-10 whether reply(prefix) works for the full turn as well as reply(full).
- RunInfra `glm-5-3-flash`, reasoning low. TB dev (annotator a, 38 convs) + oto dev (16 convs):
  16.2k candidates, $0.44.

Label reliability, 1500 random items re-judged: same model, another batch, no context: r 0.78,
binary agreement 83%; `deepseek-v4-1-flash`: r 0.75, agreement 78% (stricter, 27% vs 44%
sufficient).

## What the labels say (`suff_stats.py`, score >= 8)

| | TB dev | oto dev |
| --- | --- | --- |
| candidates sufficient before the turn end | 44% (mid) / 41% (pause) | 45% / 45% |
| stable-sufficient lead before the turn end, median | 1.23 s | 1.33 s |
| turns with lead > 1 s / > 3 s | 53% / 33% | 53% / 33% |
| turns of >= 15 words, median lead | 1.95 s | 2.12 s |

The score rises with the fraction of the turn heard (4.2 at the first 10% of words, 7.9 in the
last 10%), but 23% of turns flip between sufficient and not 3+ times.

Against the "speaker is done" completion labels (semantics branch, same oto dev pauses, 983 holds):
r = 0.18. 26% of holds are sufficient but not done (the reply could be prepared while the speaker
goes on), 19% are done but not sufficient. So it is a different target from EOT/completion.

## Can it be predicted causally? (`suff_feats.py`, `suff_probe.py`, `suff_gate.py`)

5-fold CV grouped by conversation, target score >= 8 at candidates before the turn end (base 0.44).

| predictor (causal) | AUC | TB / oto |
| --- | ---: | --- |
| position: words so far, elapsed time, pause flag | 0.628 | 0.639 / 0.606 |
| streaming FastConformer frame, own + other channel | 0.601 | 0.617 / 0.561 |
| + mean of last 1 s | 0.619 | 0.631 / 0.588 |
| encoder + position | 0.620 | 0.632 / 0.589 |
| LLM gate, one call on (context, words so far), 2.6k items | 0.608 | |

Encoder frames come from the cached `feats_asr` on the Modal volume (CPU only, cents). `mid`
times are interpolated, so frame alignment there is approximate; pauses are exact and no better.

## Verdict

- In hindsight the target exists: in half of turns the right reply is fixed more than 1 s before
  the speaker stops (median 1.2-1.3 s, ~2 s on longer turns).
- Nothing causal predicts it: neither the encoder nor an LLM reading the words so far beats
  the trivial position baseline (AUC ~0.6), well under what the label noise allows. Whether
  the speaker is about to add something that changes the answer is not visible in the past,
  at least in this casual chit-chat, where replies are generic. Task-oriented speech (questions,
  requests) may behave differently; not tested.
- So a "sufficiency head" is not worth training. What the lead time does support is speculation:
  run the LLM on the transcript at every pause or clause boundary (prefill + draft, cheap), keep
  the latest draft, and let the EOT policy decide when to commit. In half of turns the draft is
  already right when the commit comes. Interruptions mid-turn should stay with the LLM, gated
  high, not with a turn head.

## Speculation: hit rate and stale-draft detection (`suff_spec.py`, `suff_rescore.py`)

Rin's follow-up: speculative drafting makes sense, so how well does it work, and can we tell live
whether the current draft is still right? Every candidate's draft was judged against the full turn,
so keeping the draft from candidate j until the commit is a hit when its score is >= 8. The final
pause always yields a fresh full draft; speculation only matters when the LLM is slower than the
EOT delay, so these are hit rates for the last draft started before the speaker's final pause.

| redraft at | turns with a draft point before the final pause | hit rate there | lead before turn end (median) | drafts per turn |
| --- | ---: | ---: | ---: | ---: |
| VAD pauses only | 36% | 54% | 5.6 s | 1.7 |
| pauses + clause boundaries | 80% | 65% | 1.5 s | 5.6 |

So with clause-level redrafting about half of all turns (80% x 65%) have a correct reply ready
before the speaker's final pause.

"Is the current draft final?" at 10.8k candidates with a previous draft (base rate 0.47):

| causal signal | AUC |
| --- | ---: |
| words so far | 0.602 |
| word overlap of consecutive drafts | 0.532 |
| LLM similarity of consecutive drafts (0-10, $0.05) | 0.576 |
| re-scoring: log-prob change of the previous draft under the new prefix (Qwen2.5-1.5B, L4, 4k items) | 0.507 |
| all combined, 5-fold grouped logistic | 0.624 |

A draft that stopped changing is final 54-56% of the time vs 42-46% otherwise. Neither stability
nor re-scoring tells us whether the cached draft will survive, matching the earlier result: what
the speaker says next is not predictable from what came before in this data.

Practical consequence: redraft at every clause boundary (cheap with a prefilled cache), always
start a fresh draft at the final pause onset, and treat an older draft as a ~65% guess. Don't
verify it; use it where being wrong is cheap (prefetching tools or retrieval, warming TTS, a
lead-in or backchannel) and speak the fresh draft once it is ready.

Cost: sim $0.05 (gateway), re-scoring ~$0.05 (Modal L4).

## Reproduce

```bash
pip install 'modal[api-proxy-support]' httpx scikit-learn
cd ssl_turn/pipeline   # semantics branch (PR #14) for dump_annotations.py and complete_labels.py
modal run ../semantics/dump_annotations.py --out ann.json            # raw labels, never commit
python ../sufficiency/suff_labels.py ann.json suff_dev.json          # ~$0.44
python ../semantics/complete_labels.py ann.json comp_dev.json        # ~$0.02, oto dev only
python ../sufficiency/suff_stats.py suff_dev.json comp_dev.json
modal run ../sufficiency/suff_feats.py --labels suff_dev.json --out suff_feats.npz
python ../sufficiency/suff_probe.py suff_dev.json suff_feats.npz
python ../sufficiency/suff_gate.py ann.json suff_dev.json             # ~$0.02
cd ../sufficiency
python suff_spec.py suff_dev.json --make-sim sim.json                 # ~$0.05
modal run suff_rescore.py --ann ann.json --labels suff_dev.json --out rescore.json
python suff_spec.py suff_dev.json --sim sim.json --rescore rescore.json
```
