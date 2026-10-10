# ssl_turn in the browser (onnxruntime-web, WebGPU / WASM)

Streaming export of the deployable turn model: the causal FastConformer encoder
(`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`, att context [70, 0], 109M params)
plus a `feats: asr` head (r019 `bgaug_s1`, 3M params), run one chunk of 80 ms frames at a time.

| file | what |
|---|---|
| `stream_encoder.py` | `StreamStep`: raw 16 kHz audio window -> K frames of 1024-d features, with NeMo cache-aware caches. Mel front end in-graph. |
| `stream_head.py` | `HeadStep`: the TurnModel head with KV caches (20 s ALiBi window); outputs floor posteriors, p(SILENT), fine labels and the `eot_q` / `int_nobc` score tracks. |
| `export_onnx.py` | Exports both, plus encoder variants, and checks each against PyTorch. |
| `js/turn_stream.js` | Browser runtime: `TurnStream.push(ch0, ch1)` per 80 ms block. |
| `playground/` | Live two-speaker demo (mic, audio files or a stereo file, optional background audio) with the TurnBench commit policy; `build.sh` assembles the static Hugging Face Space ([shinohararin/turn-playground](https://huggingface.co/spaces/shinohararin/turn-playground)). |
| `bench/` | `bench.html` (streams a clip, checks every frame against Python), `make_ref.py`, `run_chromium.mjs` (headless here), `modal_webgpu.py` (Chrome + WebGPU on a Modal L4). |

```
python ssl_turn/web/export_onnx.py --head r019_asr_bgaug/bgaug_s1.pt --out web_models --frames 1,2,4
```

## Exactness

The streaming encoder reproduces `encode_asr.py`'s offline features: max relative error 4e-4
against the offline fp32 forward, which is the fp16 rounding of the stored features (the cached
GPU features in `feats_asr` are themselves 2% off fp32 because of bf16). Two details matter:

- NeMo's own `cache_aware_stream_step` is **not** equal to the offline forward at the start of a
  stream: its first chunks feed zero mel frames through the subsampling convs as data, and
  `ReLU(bias)` leaks into the attention caches for ~80 s (relative error ~1.0). `StreamStep`
  zeroes pre-start frames at every causal conv input, and replicates `torch.stft`'s reflect
  padding at sample 0.
- Encoder frame k sees mel frames up to 8k (audio up to 1280 k + 256), so our frame t (encoder
  frame t - 1) is computable from audio up to 1280 t - 1024: 80 ms earlier than the
  `(t + 1) * 80 ms` the TurnBench scorer assumes.

The head step matches the offline head to 1e-6, for K = 1, 2 and 4 frames per call. In Chromium,
fp32 WASM and WebGPU both match Python to 1e-6 on 60 s of TurnBench dev audio; `w16` to 7e-4.

## Variants (per K)

| variant | size | math | use |
|---|---|---|---|
| `encoder_kK.onnx` | 424 MB | fp32 | reference |
| `encoder_kK_w16.onnx` | 213 MB | fp16 weights, fp32 math | **default**: half the download, runs on any backend |
| `encoder_kK_fp16.onnx` | 213 MB | fp16 | WebGPU adapters with `shader-f16` only; without it ORT falls back to CPU kernels (14x slower) |
| `encoder_kK_int8.onnx` | 149 MB | dynamic int8 matmuls | not worth it: posterior error 0.04-0.05 and slower than w16 on WASM (RTF 0.96 vs 0.69 at K = 4) |
| `head.onnx` | 12 MB | fp32 | K is a dynamic axis |

## Speed (Chromium, onnxruntime-web 1.22.0, 60 s of TB dev audio, both channels)

Real-time factor = time per call / audio per call (80 ms x K). Head on WASM in all rows.

| backend | K = 1 | K = 2 | K = 4 |
|---|---|---|---|
| WebGPU, NVIDIA L4 (Modal, Vulkan, w16) | 0.77 (61 ms) | 0.37 (60 ms) | **0.19** (62 ms) |
| WASM, 4 threads, this cloud VM (w16) | 2.2 | 1.04 | 0.69 |
| WASM, 4 threads, Modal CPU (w16) | | | 0.57 |
| native onnxruntime CPU, 4 vCPU (fp32) | 1.1 | | |

On WebGPU the call costs ~47 ms (encoder) + ~12 ms (head on WASM) whatever K is: it is
per-op dispatch overhead over ~1500 graph nodes, not GPU compute. So K = 2 or 4 is what makes it
comfortable. K adds latency only to the earlier frames of a chunk (up to (K - 1) x 80 ms), and
with K = 2 the first frame of each chunk still lands at the scorer's assumed time because of the
80 ms head start above. On CPU the encoder is weight-bandwidth bound (213-424 MB of weights per
call), which K also amortises.

Not measured here: a laptop's integrated GPU, Apple Silicon, phones, and `shader-f16` (Chrome on
the L4's Linux driver does not expose it). WebGPU in this container is SwiftShader (CPU), which
is only good for correctness.

## Ideas not done

- WebGPU graph capture (`enableGraphCapture`) or fewer nodes (fixed-shape simplification) to cut
  the ~47 ms fixed cost; int4 `MatMulNBits` weights for a ~70 MB download.
- The `turn-1-mini` style Silero VAD + policy runs in a few ms per second of audio on WASM; it
  is not part of this export.
