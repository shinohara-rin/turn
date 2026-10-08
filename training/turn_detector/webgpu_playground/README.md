---
title: Pardon Turn Detector
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
header: mini
short_description: Realtime end-of-turn detection in your browser with WebGPU
models:
  - nvidia/parakeet_realtime_eou_120m-v1
tags:
  - audio
  - webgpu
  - onnx
  - turn-detection
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---
# Pardon Turn Detector — browser playground

[Open the full-screen playground](https://shinohararin-pardon-turn-webgpu.static.hf.space/).

A public research preview of the continued-full17 Pardon end-of-turn detector. All inference is on the visitor's device. This static Space uses no server GPU and has no audio-upload endpoint. Desktop Chrome or Edge with WebGPU is required. Load the FP32 model (~466 MB plus the browser runtime), then use the microphone, the generated spoken example, or a local audio file. Model assets may be cached by the browser.

## Input modes

**One speaker:** the first audio channel is the target; the companion channel is actual zero-valued audio. The companion still passes through the encoder and Silero frontend. Its embeddings, activity history and log-energy are not replaced with all-zero features. Only the target's events are displayed. This is an inference baseline, not a model trained or quality-qualified specifically for mono input. It still uses the batch-two encoder and does not halve encoder computation.

**Two speakers:** one speaker must occupy each stereo channel. This is not diarization or source separation. A mixed mono recording cannot be interpreted as two isolated speakers.

## Streaming implementation

16 kHz Web Audio capture in 2560-sample (160 ms) blocks; pinned 128-bin log-Mel frontend with 512-point centered STFT and explicit right-context availability; independent recurrent encoder caches; selected 1042-input MLP head with its original training normalizer; five 32 ms Silero probabilities averaged per decision. The fixed serving policy uses threshold 0.8214424509124978, speech gate 0.5 and 1.5 second recommit. Encoder features retain the original float16 cache-rounding step before the FP32 head. Model arithmetic is FP32; the rejected full-FP16 encoder is not used.

The encoder uses ONNX Runtime Web 1.30.0 WebGPU with its large recurrent caches retained on GPU. Head and VAD run on local WASM. CPU fallback may occur inside the encoder. Availability timestamps, execution time and wall-clock lag are distinct. The UI shows lag and stops if the queue exceeds 4.8 seconds; no audio is silently dropped to pretend that slow hardware is keeping up.

Stop/start resets conversation state. The generated eSpeak example contains no dataset audio. The optional device check uses deterministic synthetic noise/silence and compares recurrent encoded features, VAD and head probabilities against an ORT CPU reference. It tests implementation consistency, not real-speech accuracy. Browser resampling, device processing and compute lag can differ from benchmark conditions. This is not a TurnBench result or a claim of parity with Ooma.

## Provenance and licenses

The browser deployment is a separate copy authorized for public release. The original research archives remain private. `assets/manifest.json` records deployment hashes. Encoder export revision: `d400823b62a3170cb2aaa1c56871848b9622f71d`; selected head SHA256: `cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40`.

The encoder is a continued/converted derivative of [NVIDIA Parakeet realtime EOU 120M v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1), governed by the NVIDIA Open Model License. This does not relicense those weights. Silero VAD is MIT licensed. ONNX Runtime is MIT licensed; bundled notices are in `licenses/`. No original dataset, training audio, evaluation labels, test predictions, optimizer state or credentials are included in this Space.
