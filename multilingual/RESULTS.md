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
