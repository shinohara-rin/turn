# Frozen encoder turn heads

Run training only on the remote Colab runtime. No dataset is fetched by this
module. `heads.py` consumes the parent's causal feature caches; cache causality
and split isolation remain upstream responsibilities.

```bash
python experiments/turn_detector/heads.py train --manifest /content/cache/manifest.json --out /content/runs/mlp-seed42 --variant mlp
python experiments/turn_detector/heads.py train --manifest /content/cache/manifest.json --out /content/runs/vap-seed42 --variant vap
python experiments/turn_detector/heads.py evaluate --manifest /content/cache/manifest.json --checkpoint /content/runs/mlp-seed42/best.pt --split dev --out /content/runs/mlp-seed42/dev.json
```

Each output directory must be new. Configurations, every epoch checkpoint,
all threshold/recommit scores, selected operating point, and best checkpoint
are retained. Selection maximizes official aggregate EOT recall under 0.1
negative-span FPR, then prefers lower FPR/latency. No qualifying point remains
explicitly over the ceiling rather than being presented as qualifying.

Inputs: manifest list of `{id, split, npz, events}`; paths relative to manifest
(or absolute). `times [T]`, `features [T,2,D]`, `vad [T,2]`, optional
`extras [T,2,F]`, and training-only `annotation_activity [T,2]`. Events JSON
uses official `ConversationEvents` keys and dataclass field names. VAP auxiliary
targets use only training annotation activity, never prediction inputs.

Positives start at gold EOT and stop at 2.5 s or the next detected own-channel
speech onset. Resume is currently inferred from cached VAD transitions, since
the event sidecar does not contain full speech segment starts. This may add
label noise; true annotation segment starts can refine supervision upstream.
Explicit negative pause windows and positives are normalized per window;
background quiet is normalized per episode and downweighted. Initial listening
before any detected own speech receives no background weight. Gold exclusion
intervals receive no EOT loss weight.

The deployment policy fires while own VAD < 0.5 after observed own speech,
once per episode, optionally repeating no sooner than 1.5 s. Recommit policy
and thresholds are selected on dev. Fixed decision availability grid is
160 ms; no retrospective frame timestamps are emitted. Gate evaluation requires
`--split gate --allow-gate` and uses the checkpoint's frozen operating point.
This explicit switch is an authorization guard, not a substitute for upstream
permission to unseal the gate.

Synthetic checks (no training/data):
`uv run --with numpy python experiments/turn_detector/test_heads.py`.
Only MLP and VAP variants are implemented; temporal GRU is a future hypothesis.

## Prespecified silence baselines

`baseline.py` reads only times, VAD and event sidecars from dev caches, so encoder
features are unnecessary. The 72 fixed policies cross nine own-silence durations,
four other-speaker gates (none / 160 / 320 / 640 ms sustained speech), and the
two existing recommit policies. All decisions remain causal. It reports overall
and per-family best points; no gate data is opened during selection.

```bash
python experiments/turn_detector/baseline.py sweep --manifest /content/cache/manifest.json --out /content/runs/silence-baselines
python experiments/turn_detector/baseline.py evaluate --manifest /content/cache/manifest.json --policy /content/runs/silence-baselines/best.json --split dev --out /content/runs/silence-baselines/dev-frozen.json
```

The output preserves all attempts, grid, manifest/scorer hashes, and dev IDs.
`evaluate` performs no sweep and requires explicit authorization flag for gate.
Outputs may not overwrite earlier runs. Tests include initial-listening behavior,
causal sustained-speech gating, protected split file access, and episode weight
invariance when background silence is extended tenfold.

## Atomic epoch recovery

Both `heads.py train` and `temporal_head.py train` now accept `--resume` (latest
numbered epoch in the existing output directory) or `--resume /path/epoch-003.pt`.
Repeat the same training command and hyperparameters, adding `--resume`:

```bash
python experiments/turn_detector/heads.py train --manifest /content/cache/manifest.json --out /content/runs/mlp-seed42 --variant mlp --resume
```

Every epoch is saved immediately after its training phase, then saved again
after dev scoring. `epoch-NNN.pt` and `latest.pt` include optimizer, Torch CPU
and CUDA RNG, NumPy RNG (including the GRU Generator), fitted scaling, epoch,
all dev attempts, best selection, and a copy of the best inference model.
Interrupted dev scoring is repeated without repeating that epoch's training.
Attempts and best artifacts are reconstructed transactionally on recovery;
resuming a completed run adds no epoch rows. An interrupted training epoch
restarts from the last saved epoch boundary, not its last minibatch.

Writes use a hidden temporary `.tmp` file, flush/fsync, then atomic replacement.
Backup processes should mirror finalized `*.pt`/JSON outputs and ignore `.tmp`.
Point `--out` at persistent storage or use the parent's remote backup daemon;
atomic local writes alone do not persist beyond VM loss. Restoring artifacts
into a new output directory can use an explicit resume checkpoint path.

Hyperparameters, total epoch budget and manifest content must match the saved
run; paths may change after moving to a new VM. `best.pt` remains the compatible
inference artifact. Older checkpoints lacking optimizer/RNG remain evaluable,
but exact training resume correctly rejects them. Resume from the new numbered
epoch or `latest.pt`, not the inference-only `best.pt`.

Pure file/config tests run locally without torch. Full interrupted-versus-
uninterrupted synthetic CPU checks are opt-in on the remote runtime:

```bash
RUN_CHECKPOINT_TORCH_TESTS=1 python -m unittest discover -s experiments/turn_detector -p test_checkpointing.py
```
