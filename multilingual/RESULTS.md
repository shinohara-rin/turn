# Multilingual encoder probe (es / ja / zh)

Question (2026-10-10): is there a streaming multilingual encoder about the size of our r019 backbone
(`stt_en_fastconformer_hybrid_large_streaming_multi`, 109M encoder) that keeps end-of-turn cues
in Spanish, Japanese and Chinese? Bigger is ruled out for client-side real-time inference.

## Setup (`encoder_probe.py`)

- Data: LiveKit eot-bench (`livekit/eot-bench-data` @ `ca9d98a9`), single user turns. Every silence
  span >= 0.2 s is a readout, 0.2 s into the span; the final span is the true end (1), earlier
  spans are holds (0). en 1015 readouts / 400 ends, es 995 / 400, ja 947 / 356, zh 869 / 344.
- Features from the last 16 s of audio cut at the readout (nothing after it is seen, whatever the
  encoder's lookahead): final frame + mean of the last 1.04 s, per tap layer.
- Linear probe: standardize, PCA 256, logistic regression (C by inner grouped CV). "Within" = 5-fold
  CV grouped by turn inside one language. "en>X" = trained on English, tested on language X (only
  for encoders that cover all languages).
- EOT only. eot-bench has no second speaker, so interruptions are not tested.

## Results (ROC AUC, holds vs. true ends)

| encoder | encoder params | en | es | ja | zh | en>es | en>ja | en>zh |
|---|---|---|---|---|---|---|---|---|
| **fc_en (r019 backbone, English)**, L17+L9 | 109M | **0.950** | 0.941 | 0.980 | 0.896 | 0.925 | 0.909 | 0.855 |
| Nemotron 3.5 bottom 4 layers, L4+L2 | ~105M | 0.894 | 0.934 | 0.968 | 0.900 | 0.889 | 0.898 | 0.857 |
| Nemotron bottom 6, L6+L4 | ~155M | 0.920 | 0.943 | 0.979 | 0.903 | 0.898 | 0.915 | 0.873 |
| Nemotron bottom 8, L8+L4 | ~205M | 0.919 | 0.948 | 0.979 | 0.900 | 0.902 | 0.919 | 0.860 |
| Nemotron bottom 12, L12+L6 | ~305M | 0.948 | **0.955** | **0.987** | **0.922** | 0.934 | 0.941 | 0.890 |
| Nemotron full 24, L24+L12 | 609M | 0.948 | 0.954 | 0.984 | 0.919 | **0.942** | **0.952** | **0.892** |
| Moonshine streaming, one model per language (small en/es/ja, tiny zh), L10+L5 | 51M (zh 9M) | 0.929 | 0.943 | 0.987 | 0.897 | – | – | – |

Bottom-k Nemotron = the full model's layer-k output (identical to a model truncated to k layers).
Parameter counts for truncations are estimates (~25M per layer plus subsampling).

## Reading

- **Nothing at ~110M beats the English encoder we already use.** Within each language it ties the
  bottom-4 Nemotron and Moonshine (es 0.94, ja 0.98, zh 0.90) and wins on English. End-of-turn cues
  in a single speaker's pause are largely acoustic/prosodic, and the English model keeps them.
- **Cutting Nemotron down doesn't work.** At 4 layers it is worse than fc_en everywhere, including
  English (0.894). The multilingual gain only appears at 12+ layers (~300M).
- **What a big multilingual encoder buys is modest:** +0.01-0.03 within-language (largest on zh) and
  +0.02-0.04 on English-to-other transfer. Those are the gains a same-size distilled student could
  aim at.
- Moonshine needs a separate model per language and its zh model is tiny. It is not worth switching to.
- Caveats: one readout time (0.2 s), EOT only, ~1k readouts per language (differences under ~0.01
  are noise); ja is near ceiling on this set.

## Recommendation

Keep the r019 English FastConformer as the backbone. Train the head on multilingual data (CallHome
es/ja/zh) and measure on eot-bench es/ja/zh. If the gap to Nemotron-12 persists after that, distill
Nemotron's layer 12 into a 109M student with fc_en's architecture (initialised from fc_en, untranscribed
audio only), so the browser export stays unchanged.

Cost: ~$0.5 Modal (6 L4 containers, ~5-8 min each). Features: `/mnt/project-files/multilingual/encoder-probe/`.

# Training with CallHome es / ja / zh (r020)

## Data (`callhome_modal.py`)

`talkbank/callhome` spa / jpn / zho: mono telephone calls, ~10 min annotated each, utterance timing and
speaker only. 301 calls written (es 104 / 15.6 h, ja 112 / 17.3 h, zh 85 / 12.0 h); 99 skipped where a
third speaker holds > 5% of the speech. Per call: utterances intersected with Silero VAD on the mix (so
pauses inside an utterance are silences), rule labels (`rule_labels.py`: Turn / Backchannel by timing)
through the pinned TurnBench gold builder, exactly as oto's labels are built, and pseudo-stereo
(`pseudo_stereo.gated_stereo`) encoded per channel with the r019 FastConformer.

Rule labels checked against oto's human labels (40 oto conversations, labels stripped and rebuilt
from timing): EOT events precision 0.62 / recall 0.79 at 0.3 s; backchannel precision 0.72. Longer
backchannel limits trade recall for precision without improving both.

## Training

r019 recipe (fine1_bal1: dim 192, 3 layers, 1000 steps, batch 64, all 131 oto train conversations),
plus 80 CallHome calls per language with 30% of every batch drawn from them. Three arms in one run,
two seeds each, all on the same batches:
- **ctrl:** CallHome rows get zero weight (oto only, same oto crops).
- **va:** CallHome rows train only the VAP voice-activity projection.
- **rule:** CallHome rows train every target from the rule labels.

## Results

LiveKit eot-bench, harness metrics (mean latency at 5% / 10% false-cutoff budget, lower is better;
AUC), mean of 2 seeds:

| arm | en | es | ja | zh |
|---|---|---|---|---|
| ctrl (oto only) | 829 / 502 ms, 0.969 | 843 / 627 ms, 0.941 | 759 / 543 ms, 0.958 | 889 / 618 ms, 0.917 |
| va (activity only) | 865 / 538 ms, 0.967 | 803 / 602 ms, 0.935 | 762 / 570 ms, 0.942 | 923 / 623 ms, 0.899 |
| **rule** | **812 / 472 ms, 0.970** | **782 / 562 ms, 0.947** | **700 / 531 ms, 0.953** | 885 / 623 ms, 0.917 |

Both rule seeds beat both control seeds at 5% in es (809, 756 vs 857, 829 ms) and ja (707, 693 vs
765, 753 ms). zh does not move. Per-seed numbers: `/mnt/project-files/multilingual/r020/`.

For reference, the published systems on eot-bench @ 5%: es LiveKit v1 642, Soniox 800, Deepgram Flux
820 ms; ja LiveKit v1 321, ultraVAD 462, GPT Realtime 2 736 ms; zh LiveKit v1 799, Soniox 886 ms.
The rule arm moves es from 4th to 2nd (ahead of Soniox and Deepgram Flux); ja stays 3rd, now ahead of GPT Realtime 2 by a wider margin but behind LiveKit v1 and ultraVAD.

TurnBench dev (sweep at FP <= 0.10, `eot_q@r0.5+rc1.0` / `int_nobc@r0.5+rc1.0`, recall per seed):

| arm | EOT | INT |
|---|---|---|
| ctrl | 0.941 / 0.934 | 0.983 / 0.983 |
| va | 0.934 / 0.937 | 0.986 / 0.986 |
| rule | 0.934 / 0.929 | 0.986 / 0.986 |

English TurnBench is unchanged within seed noise.

## Reading

- **Rule labels help; voice activity alone does not.** The VAP-only arm is flat or worse, so the
  multilingual signal has to reach the floor heads.
- Gains are ~60 ms at the 5% budget in es and ja, with no TurnBench cost, despite labels that agree
  with oto's annotators only 62% of the time on EOT. zh (the smallest set, 12 h) shows nothing yet.
- Next levers: more calls (CallFriend es/zh/ja on TalkBank, RAMC for zh), better labels than timing
  rules (ASR + LLM labeler, validated on oto first), and the eot-bench text cues LiveKit v1 uses.

Cost: ~$5 Modal (CallHome prep ~4.5 L4-hours, training ~10 min A100-80GB, eot-bench scoring).
