# Handoff notes (attempt 2, `ssl_turn/`): read this first

Context for continuing this work in a fresh conversation. [`README.md`](README.md) holds the
design. [`RESULTS.md`](RESULTS.md) holds every experiment with numbers. This file holds what
lives in neither: the user's goals and preferences, how to operate the environment, the
current state, and the open threads. Last updated 2026-10-09.

## What the user wants (in their words, condensed)

- **Two attempts.** A second, parallel attempt at TurnBench next to the Ooma reproduction
  in `training/`. It uses an internet-scale pretrained audio encoder (MOSS family), not SSL
  from scratch.
- **Objective:** model **floor ownership**: HELD_0 / HELD_1 / OPEN / CONTESTED.
  - Overlap is a first-class CONTESTED state, not neutral p(speaking).
  - Backchannels do not contest the floor.
  - Hand-offs are soft, interpolated targets.
  - Predict TurnBench's label structure, not a bare p(EOT). "Make full use of labels."
- **Diarization.** Wanted as an explicit objective: arrival-order slots for mono, mixed-speaker
  input. The user expects it to help the representation and to be useful to end users.
- **Later goals:** multilingual and code-switching. Not started.
- **Data:** the user is interested in podcasts made usable through separation/diarization
  (DuplexChat), with heterogeneous multi-source training. ASR+LLM pseudo-labels were also
  floated. TTS-synthesized TurnBench data was tried by the user and felt unnatural; postponed.
- **Working style:**
  - Run autonomously; small pilots first; evaluate even bad models; probe, find problems,
    iterate. Be creative and also use prior research.
  - Then hyperparameter search and ablations.
  - **GPU efficiency matters a lot** ("an A100 at 10 GB VRAM and 20% utilization feels like
    a crime"). Billing is per second, so optimize any training or inference.
- **Budget:** the original ~$20 Modal allocation was spent. The user then granted **~$12 more**;
  about **$4.2 of that is used** (Modal cost on 2026-10-09 was $4.81, of which $0.63
  predates the grant). LLM calls go through the user's RunInfra gateway; that spend is not
  tracked here. Keep pilots small and report cost.
- The user is not certain about the Cat encoder and asked about the MOSS-Transcribe-Diarize
  (MTD) backbone. Both are implemented; see RESULTS.md for the comparison.

## Hard rules

- **Git:** develop on branch `claude/zen-knuth-2ni5ia` and push there. Its open PR is
  https://github.com/shinohara-rin/turn/pull/3, and pushes update it; don't open another.
  - Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_011gP2Q44dDcHhM4ZGMrsnAM`.
  - No model IDs in commits.
- **Data use:**
  - Never use TurnBench **test** audio, labels or feedback (`training/turn_detector/PROTOCOL.md`).
  - TurnBench dev is development data and is used for reporting. Thresholds picked on oto
    dev do not transfer to TB dev; report "FP at fixed recall" and seed noise.
- **Never commit** datasets, transcripts, weights, LLM outputs or credentials. Transcripts
  and LLM answers live in the scratchpad and on the Modal volume (`/work/llm/`) only.
- **otoSpeech license:** speaker identification is prohibited; diarization within a
  recording is allowed. Frozen actor split: 131 train / 16 dev / 20 gate / 253 excluded
  (seed 20261007, reproduced exactly by `prep.make_split`).

## Operating the environment

- **Modal:** account `shinohara-rin`.
  - Apps are named `ssl-turn-*`; volumes are `turnbench-datasets` (read) and
    `ssl-turn-work` (mounted at `/work`).
  - In the cloud container the CLI needs `pip install "modal[api-proxy-support]"` (for
    python-socks) and `export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt`.
  - The previous session kept a venv at
    `$SCRATCH/venv` with modal, torch (CPU), transformers 5.19, httpx and the pinned
    turnbench (`$SCRATCH/tb/src`). The scratchpad does not survive; recreate it.
- **Running scripts:** run from `ssl_turn/pipeline/`.
  - Training: `modal run train.py --run rNNN --configs configs/X.json --n-train 131 --steps 1000 --gpu H100`
    (`--no-infer` for timing pilots).
  - Scoring:
    `modal run score.py::main --run rNNN --variants eot_q,int_nobc --refractories 0.5 --recommits ",1.0"`.
    score.py has several entrypoints, so `::main` is required.
  - Stopping an app: `modal app stop -y ap-...` (needs `-y` without a TTY).
- **Cost:** `modal billing report --for today --resolution h --show-resources --json`;
  sum the rows whose description starts with `ssl-turn`.
- **RunInfra LLM gateway:**
  - `https://api.runinfra.ai/v1/chat/completions`, model `nemotron-3-5-lightning-30b`.
  - The egress proxy injects auth, so **send no Authorization header** (the env var is
    empty).
  - Thinking runs away: use `reasoning_effort: "none"` and ask for short visible reasoning
    in the content instead. About $0.0001 per short call.
- **Volume layout (`/work`):**
  - `split.json`, `extra_no_gate.json`;
  - `audio/{oto,tbdev}`, `labels/oto/*.npz` (including `fine`), `gold/oto`, `gold/tbdev.json`;
  - `feats/{oto,tbdev}/{cid}.npy` (Cat, fp16 [T, 2, 5888] = taps 7/15/23/31 × 1280 + final 768);
  - `feats_mtd/` (MTD 4096-d; TB dev plus a 23-conversation oto subset only);
  - `models/cat/` (pinned code + encoder-only weights);
  - `runs/<run>/{probs.npz, *.pt, train.json, score_*.json}`;
  - `llm/` (transcripts and LLM outputs; never commit).
- **Tests:** `CAT_CODE_DIR=<dir with pinned Cat code> python -m unittest test_cat_top test_cat_encoder`,
  and so on. The turnbench pin conflicts with recent transformers, so keep it in its own
  environment or path.

## Current state (TurnBench dev; details in RESULTS.md)

- **Best system:**
  - Model: frozen Cat taps 15/23/31 + final, into the floor-ownership head (r012
    `fine1_bal1`: fine 18-class head, inverse-frequency balanced).
  - Scores: `eot_q` = (p(OPEN) + p(HELD_other)) × p(SILENT); `int_nobc` = int_spk × (1 −
    own backchannel + noise mass).
  - Commit policy: 0.5 s refractory, EOT re-commit after 1 s.
  - Official dev operating point (FP ≤ 0.10; r004 file `predictions-dev-rc.json`): EOT
    0.939 / FP 0.068 / p50 296 ms, INT 0.974 / FP 0.085 / p50 585 ms. Seed noise is about
    ±0.006 (EOT) to ±0.02 recall.
- **The model is near the label ceiling** (`score.human_ceiling`):
  - Against gold built from the other two annotators, a single annotator gets EOT
    0.82–0.89 recall at FP 0.09–0.12; the model gets 0.937 at 0.080.
  - Remaining EOT false fires are mostly in pauses that all three annotators call holds.
  - INT false fires (backchannels) are never labeled as interruptions by humans. They are a
    causal-decision problem: at ~400 ms a "yeah" sounds like a take-over.
- **Saturated or negative (do not repeat without a new idea):**
  - Head size, taps, loss weights, label smoothing, more otoSpeech data (32 → 231
    conversations): all within noise.
  - Cat+MTD fusion did not help. MTD alone matches Cat's recall and fires ~150 ms earlier,
    but needs an expensive encode.
  - LLM text oracle, two designs:
    - The fusion gain was a timing leak; a shuffled-answer control matched it.
    - A leak-free verifier on the model's own fires was worse than random gating.
    - The user concluded that bolting an LM on this way is a dead end.
  - LoRA fine-tuning of Cat top layers 16–31 (r013/r014): ~1% dev loss, no TB gain, and
    overfits after ~500 steps on 131 conversations.
  - Confirmation window (score persistence):
    - The INT gain did not survive a split-half check.
    - EOT: a 160 ms hold gives +~0.015 operating-point recall on both TB halves, but not on
      oto dev. It is a knob, not a default.

## Open threads / next options (user's latest choice pending)

1. **More, different data (recommended next; the user's original interest):** the podcast
   pipeline in README ("Podcasts", "Mixed-source training"; `podcast_subset.py`,
   `pseudo_stereo.py` exist). Start with a small separation-quality pilot (calibrate on
   otoSpeech mono mixes) before spending budget.
2. **Prosody features** (F0, energy, rate per 80 ms) next to Cat features, aimed at EOT
   false fires in unanimous holds. About a $1 run.
3. Backbone: a causal MTD student, or distill MTD into the Cat head (MTD is ~150 ms faster).
4. Mono / diarization (slot) path, and multilingual: designed in README, not trained yet.

## Gotchas learned the hard way

- **Cat:** needs TF32, not bf16, for the full encode (bf16 drifts to cosine 0.88). The
  upstream ring KV cache drops keys for multi-token chunks; `CatEncoder._widen_caches` fixes
  it.
- **CatTop:** re-running from cached tap 15 is exact, because RoPE is relative.
  - Training crops need 125 warm-up frames.
  - Inference context = head context + 16 × 124 frames.
  - Load only the cached columns a run reads; loading all taps OOMed an H100.
- **Training memory:** features live in VRAM; preallocate (`torch.cat` doubles peak). Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already set in `train.py`).
- **Volume I/O:** commit in batches. Per-item commits idled an H100 for ~30 s each.
- **TB dev decode:** each CPU container read the whole 4.2 GB parquet. Parallelize by
  item, not by container.
- **Selection:** thresholds from oto dev don't transfer to TB dev, and seed noise is large.
  Compare with FP at fixed recall over ≥2 seeds; check split halves before claiming a gain.
- **LLM oracle:** any query timed on annotated boundaries leaks gold timing; always include a
  shuffled-answer control. ASR lag (0.3 s) exceeds the audio model's EOT latency (~150 ms).
- **`train.py::infer`** references an undefined `use_dev`. Fix it before using it.
