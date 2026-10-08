# Pardon turn-detector training

Source for the independent Ooma-style recreation and its browser deployment.
The selected model is a VAP-continued Parakeet encoder plus a paired MLP head.
This is a research pipeline, not a claim to reproduce Ooma's unpublished weights.

[Public browser playground](https://huggingface.co/spaces/shinohararin/pardon-turn-webgpu)

## Layout

- `turn_detector/`: ingestion, actor split, causal audio features, encoder continuation, head training, event scoring, checkpoint recovery, focused tests and export.
- `turn_detector/single_speaker/`: bounded frozen-baseline / stereo-control / mono-adaptation comparison. Its scripts restore approved private Drive caches; they are not a generic fresh-data launcher.
- `turn_detector/webgpu_playground/`: static browser UI, audio worklet, frontend, ONNX worker and remote deployment scripts.
- `source-snapshot.json`: original source commit and per-file hashes. Some included experiments were work in progress when first synced.

## Where to run

The scripts use the historical Colab layout (`/content/turn-recreation`, `/content/turnbench`,
`/content/hf`). Any host that provides that layout works; the HF token comes from the
`HF_TOKEN` environment variable (preferred) or `/content/.hf_token`.

- **Colab / any GPU machine (SSH, cloud VM):** copy `turn_detector/` to
  `/content/turn-recreation/` and run `python setup_remote.py --install-deps`. Off Colab,
  `/content` may not exist: set `TD_WORKSPACE=/path/to/workspace` (symlinked to `/content`,
  needs permission to create it) or create the directory/symlink yourself first. Export
  `HF_TOKEN`. Dataset caches fall back to per-conversation HF downloads when no pre-staged
  copy is mounted, so Drive is optional.
- **Modal:** `turn_detector/modal_app.py` builds the image from `requirements.txt` and mounts persistent
  Volumes behind the same `/content` paths. Stage datasets once with `modal_stage.py`, then run any stage command
  unchanged, from `turn_detector/` with `HF_TOKEN` exported locally:
  `modal run modal_app.py --cmd "python -m unittest test_heads"` (CPU) or
  `TD_MODAL_GPU=A10G modal run --detach modal_app.py --gpu --cmd "python cache_batch.py --batch-size 4 --output-dir /content/turn-recreation/cache-streaming-v2"`.
  `drive_backup.py` is unnecessary there (Volumes are persistent and committed every 60 s).
  Not ported: `gate_evaluate.py` (hard-requires real `/content/drive/` paths) and
  `single_speaker/` (restores private Drive caches); run those on Colab.

## Environment and data boundaries

**Run training and all data access on remote Colab. Do not download data locally.**
Use the Colab CLI; mount Drive before expensive work. No GPU is required for cached-feature head training. Encoder feature extraction/continuation benefits from a GPU; allocate one only for that stage and release it afterward.

The original scripts expect `/content/turn-recreation` and `/content/turnbench`.
On the remote VM, clone this repo and copy `training/turn_detector/` into
`/content/turn-recreation/`. Supply an existing authorized HF token in
`/content/.hf_token` with mode 0600; never commit it. Run `setup_remote.py` there.
It installs the training dependencies and pins the evaluator to
`SesameAILabs/turnbench@38a6f874322430cb3ca71d8a52aa1e636e88bad8`.
The historical setup uses dependency ranges rather than a complete lockfile;
record a fresh `pip freeze` with every run. Exact numerical reproduction across
new runtime builds is not asserted.

Pinned sources:

| Component | Revision |
| --- | --- |
| otoSpeech | `otoearth/otoSpeech-full-duplex-turn-104h@46f520297f434edf804389f82f9075a59d2f8268` |
| Parakeet | `nvidia/parakeet_realtime_eou_120m-v1@a7e2b4629593dce0ec19f600e00e9904353fda2d` |
| TurnBench evaluator | `38a6f874322430cb3ca71d8a52aa1e636e88bad8` |

The frozen actor partition contains **131 train / 16 dev / 20 gate** conversations;
253 cross-partition conversations are excluded. The pilot uses 16 train / 6 dev.
Training and normalizer fitting accept only approved otoSpeech train records.
Public TurnBench dev was used for development and calibration, so it is not an
untouched holdout. The otoSpeech gate has already been consumed by earlier work.
**No TurnBench test examples, labels, predictions, pseudo-labels or feedback may
enter training, normalization, threshold tuning or model selection.** No
submission is performed by this setup. Consult `PROTOCOL.md` for the original
protocol and amendments; historical status statements in old recipe files are
not current run status.

## Training stages

Run commands below **on Colab**, from `/content/turn-recreation`.
Mount Drive first and check that `/content/drive/MyDrive` is real. Start
`drive_backup.py` as a supervised background process; it copies checkpoints and
completed caches with SHA256 readback every 30 seconds. Its dated destination
is the existing project archive; use a new destination for an independent run.
Keep dataset staging separate from training runs.

1. `python inspect_data.py` collects remote metadata. It also attempts an encoder
   load for inspection; feature extraction itself uses pinned model revisions.
2. `python prepare_data.py --train 16 --dev 6` freezes the split and fetches only
   pilot train/dev audio remotely. The saved `split.json` is reused thereafter.
3. `python cache_batch.py --batch-size 4 --output-dir /content/turn-recreation/cache-streaming-v2`
   materializes causal features and validates identities. Increase the batch only
   after measuring actual GPU memory and throughput. Source audio may be reused
   from the verified Drive dataset staging directory.
4. `python run_pilot.py` fits the baseline and paired MLP/VAP heads. Head-only work
   can move to a cheap CPU runtime with verified caches and the same split.
5. `python continue_vap.py --manifest /content/turn-recreation/cache-streaming-v2/manifest.json --checkpoint /content/turn-recreation/runs/encoder-vap-v1/checkpoint.pt`
   performs the fixed 100-update encoder intervention on the 16 training
   conversations. Run its `--synthetic-smoke` qualification first on the GPU.
   It saves every 25 updates and resumes only compatible state.
6. Once continuation is finished, `run_continuation_pilot.py` compares its paired
   head. `run_full_scale.py` expands to 131 train / 16 dev with seeds 42 and 17
   for frozen and continued encoders. It expects the completed pilot artifacts.
   Keep existing checkpoint directories intact; head training supports `--resume`.
7. Use `benchmark_infer.py` and `score_benchmark_dev.py` only for explicitly scoped
   development evaluation. See `BENCHMARK_INFERENCE.md`; do not use test outcomes
   to revise the model. Event recall, negative-span FPR and matched-event latency
   matter; frame accuracy is not a promotion criterion.

Inspect each script's `--help` before a new run. These stages have dependencies;
this is intentionally not an unconditional all-in-one GPU launcher. Long jobs
must log progress, resource use and the latest verified checkpoint.

## Selected deployment

- Encoder checkpoint SHA256: `2cd76b022820e769ee95c6da08d075f5c30eadf70b0b414ff92a3c6c5c89489f`.
- Head SHA256: `cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40`.
- 16 kHz, two separated audio channels, 160 ms decision grid.
- 1042 features → 128 → 128 → 1 MLP, GELU, original train-fitted normalizer.
- Threshold `0.8214424509124978`, mean of five causal Silero frames, 1.5 s recommit.
- Single-speaker inference supplies real zero audio to channel two, preserving
  the encoder/VAD silence trajectory. It does not replace embeddings with zeros
  or halve batch-two compute. Dedicated mono adaptation is experimental.
- Browser export uses FP32 ONNX Runtime Web 1.30.0; full FP16 failed recurrent
  parity and was not promoted. Browser checks are implementation checks, not
  benchmark accuracy evidence.

Public-dev selected policy previously achieved recall 0.953782, FPR 0.081844,
median 375 ms. These are development results after tuning, **not a test score or
proof of matching Ooma**. Research archives/checkpoints remain in the owner's
private HF repositories; only the authorized deployment copy is public.

## Validation and licenses

Run focused `unittest` files on a remote environment with dependencies installed.
The tests cover leakage rejection, causal features, checkpoint recovery and head
contracts. Full retraining and real-audio evaluation are separate qualifications;
a source sync is not a new reproduction run.

No datasets, raw audio, derived training caches, model weights, credentials or
benchmark predictions are stored here. Upstream model/data licenses remain in
force. Parakeet weights use the NVIDIA Open Model License; Silero and ONNX Runtime
have their respective MIT notices. Browser deployment scripts refer to private
owner archives and require their access; a public visitor only needs the Space.
