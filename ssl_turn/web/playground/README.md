---
title: Turn-Taking Playground
emoji: 🎙️
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Live turn-taking model in the browser (WebGPU)
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Turn-taking playground

A streaming turn-taking model that runs in the browser with onnxruntime-web (WebGPU, or WASM on
the CPU). Give it two speakers, each from the microphone or an audio file (or one stereo file
with a speaker per channel), and it scores every 80 ms frame for each speaker:

- **end of turn**: the speaker has finished and is handing over the floor;
- **take the floor**: the speaker is starting a real turn (not a backchannel).

Events are committed with the TurnBench rule (rising edge above a threshold, 0.5 s refractory),
at the TurnBench dev operating points by default. Audio never leaves the device.

Model: causal FastConformer encoder (`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`,
CC-BY-4.0), 80 ms frames, no lookahead, plus a 3M-parameter cross-speaker turn head trained with
background-speech augmentation. The encoder is stored with fp16 weights (213 MB). English only.
Research preview for non-commercial use. Runtime: onnxruntime-web 1.22.0 (MIT).

Source: `ssl_turn/web/` in the turn repository (`playground/` is this page).
