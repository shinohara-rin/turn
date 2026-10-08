# Single-speaker adaptation screen

This experiment asks whether a short, dedicated mono adaptation helps beyond simply supplying zero audio for an absent second speaker. Read `PROTOCOL.md` before running; it freezes data access, comparisons, update budgets and event metrics.

These scripts execute only on remote Linux/Colab. They consume the existing guarded 131-train/16-dev continued-encoder cache and selected checkpoint. They do not bootstrap a new model from raw audio. No dataset or model files belong in this directory.

Remote requirements: mounted Drive at `/content/drive`, owner access to the private research archive through `/content/.hf_token`, Torch, NumPy, `onnxruntime==1.30.0`, `silero-vad==6.2.0`, `huggingface_hub`, Git. Place `prepare.py`, `run.py`, `PROTOCOL.md`, and the parent directory's `heads.py`, `leakage_guard.py`, `evaluate_phase128_pair.py` in `/content/single-speaker/`.

Run `prepare.py`, verify its `PREPARED` marker, then `run.py`. Preparation verifies original Drive hashes and frozen actor splits, stages only approved train/dev feature files, and computes the deterministic zero-audio recurrent trajectory. Training saves verified checkpoints to Drive every 50 updates. A complete run archives aggregates, sources and checkpoints privately on Hugging Face; no deployment promotion or benchmark submission occurs.

The scripts preserve both correlated speaker views in each original conversation partition. Mono means an independently silent *other* channel for each target view. Gold labels, own-speaker VAD, features and frame weights remain unchanged. The frozen baseline must replay the historical stereo development counts before continuation training starts.

Outputs live remotely under `/content/single-speaker/results` and Drive `turn-detector-recreation/runs/single-speaker-v1/`. Only aggregate reports, provenance receipts and source snapshots may be copied back locally. The source manifest hash precedes absolute-path relocation; final audit and per-file hashes verify the relocated inputs.
