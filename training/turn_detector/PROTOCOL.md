# Turn-detector recreation: experiment protocol

Target observed on 2026-10-07: Ooma Turn Detector test EOT recall 0.951,
FPR 0.082, median latency 388 ms (https://turnbench.sesame.com/).
Ranking qualification is test FPR <=0.15; dev operating-point budget <=0.10.
Report both recall and latency; exceeding recall alone is not matching all axes.
Ooma weights are unpublished. This is an independent recreation, not a weight reproduction.

## Data and isolation

All data downloads, audio, annotations, derived features, and training stay remote.
Colab session: turnbench-recreation (A100). Modal is CPU-only staging, not training.
Local repository stores source and aggregate metrics only.

otoSpeech source: otoearth/otoSpeech-full-duplex-turn-104h,
revision 46f520297f434edf804389f82f9075a59d2f8268.
206 actors, 420 conversations. Connected-component split is unsuitable:
one component contains 414 conversations. Deterministic actor shuffle seed
20261007 assigns 60%/20%/20% actors to train/dev/gate; conversations crossing
partitions are excluded. Result: 131 train, 16 dev, 20 gate, 253 excluded.
Metadata-only split selection precedes any outcome inspection.
The pinned upstream TurnBench README identifies this otoSpeech repository as its
training dataset and states it is speaker-disjoint from the benchmark (README.md
line31 at commit38a6f874322430cb3ca71d8a52aa1e636e88bad8). This is upstream
provenance, not an independent content-level deduplication claim.
Initial bounded pilot uses 16 train + 6 dev conversations, selected by stable
SHA256 ordering of conversation IDs. No gate labels/audio used during search.

Training loaders reject sources other than pinned otoSpeech and validate
conversation/speaker membership against the frozen remote split plan before
opening any arrays or annotations. An auxiliary future-activity target is
training supervision, never an input feature.

Protocol amendment (2026-10-07 JST, before any TurnBench dev inference):
public TurnBench dev may now guide model development and operating-point
selection. Earlier pilot experiments used only otoSpeech dev. This explicitly
supersedes the original reservation of TurnBench dev for final calibration;
all subsequent TurnBench dev results are development evidence, not untouched
validation or an official test score. No benchmark examples enter gradient
training or normalization fitting. TurnBench test audio is for frozen-model
inference only. Test labels are private to organizers. No test error analysis,
model selection, normalization fitting, pseudo-labeling, or training is allowed.
A held-out otoSpeech gate is not the official TurnBench test; report separately.
One gate evaluation compares a frozen selected candidate with fixed baselines.
Further exploratory use must be labeled as such rather than called untouched.

## Evaluator and causality

Unmodified SesameAILabs/turnbench commit
38a6f874322430cb3ca71d8a52aa1e636e88bad8.
otoSpeech has one annotator: use published floor construction on that annotation,
without representing it as TurnBench's three-annotator consensus.
Primary metric: event EOT recall with negative-span FPR <=0.10.
Secondary: p10/p50/p90 latency, per-conversation uncertainty, false fires/hour,
throughput, and interruption performance (not yet implemented).

16 kHz dual-channel audio; one-sided FIR resampling keeps physical delay.
Decision grid 160 ms. Centered STFT right context is charged as availability
latency, incomplete chunks discarded, latest fully available cache feature held.
Synthetic future mutation and prefix truncation tests run on actual NeMo/A100.
No whole-conversation normalization. Training feature scaling fits train only.
First pilot uses causal Silero VAD, not Ooma's described three-model ensemble.
First learned models freeze Parakeet and train paired heads; auxiliary-head VAP
is not yet Ooma's encoder continuation objective.

## Search log

Arbor state is .arbor/. Baselines, supervised MLP and MLP+future activity are
competing hypotheses; all configurations retained. Checkpoints remain remote.
No final model promoted without independent gate evidence.

## Persistence and recovery

CPU-only Modal volume `turnbench-datasets` contains pinned otoSpeech separate
channels (145.08 GB), TurnBench dev (4.22 GB), and sealed test audio (13.35 GB).
No Modal GPU was allocated. The duplicate otoSpeech combined-channel WAV is
omitted; both individual channels and annotations are retained.

Colab CPU session `turnbench-drive-cpu` stages the same pinned datasets into
`MyDrive/turn-detector-recreation/datasets`, independently of A100 experiments.
Training and benchmark repositories remain in separate directories.

The A100 copies completed features, scripts, split provenance, logs, and epoch
checkpoints every 30 seconds to
`MyDrive/turn-detector-recreation/runs/2026-10-07-pilot`.
Each copied file receives SHA256 readback before atomic publication. A completed
feature backup was independently read and checksum-verified from the CPU runtime.
`backup-index.json` records individual file checksums and timestamps. An absent
Drive mount causes a hard failure rather than silently writing to local disk.

## Public dev calibration resolution (prespecified before first public score)

The initial paired pilot uses the existing51uniform thresholds times2recommit settings. Subsequent operating-point calibration will apply the pinned upstream candidate_thetas algorithm to BOTH frozen and continued models:512quantiles of pooled positive frame probabilities, unioned with the0.01through0.99grid. The serving policy remains our quiet-gated commit_events with recommit null/1.5, not the upstream rising-edge policy. Report both search stages and their attempt counts separately. This is publicdev calibration only; no test feedback, modelweight fitting, or additional heldoutgate access. Source: pinned baselines/README.md and turnbench/sweep.py at38a6f874322430cb3ca71d8a52aa1e636e88bad8.
