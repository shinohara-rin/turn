# Source-sync validation — 2026-10-08

- Copied modules and their local import dependencies compile successfully.
- Remote Colab CPU: `test_leakage_guard`, `test_checkpointing`, `test_cache_identity`, `test_heads`: 28 tests, 25 passed and 3 opt-in Torch checks initially skipped.
- Then reran `test_checkpointing` with `RUN_CHECKPOINT_TORCH_TESTS=1`: all 5 passed, including the 3 previously skipped synthetic recovery tests.
- No dataset was opened by these checks. This validates contracts and recovery, not a fresh full training reproduction.
- Public browser revision `e9c1499706410965ef2efad851f010a27504181d`: full 25-step synthetic PCM-to-score check passed on Chrome/Apple WebGPU, maximum probability error 0.000103504, VAD error 7.15372e-8, encoded error 0.00048828125, zero tolerance violations. Generated spoken example completed 76 blocks, median processing 72.3225 ms per 160 ms block, maximum queue 2, and one detected event. These are device-specific implementation checks, not accuracy results.
- Final UI/license revision: `dfe75d5d2ad67c53ddb800cbac45c5961b4a7d0f`. Model assets unchanged; includes explicit stereo-device checks, initialization error handling and bundled upstream notices.
- Microphone hardware capture has not been exercised; the generated example uses the same AudioWorklet and worker path without recording ambient user audio.

Remote browser export environment: Python runtime with torch 2.11.0+cpu, numpy 2.1.3, onnx 1.23.2, onnxruntime 1.30.0, silero-vad 6.2.0, huggingface-hub 1.33.0. This is the export/check environment, not the historical training environment.
