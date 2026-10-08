# Causal GRU head vs MLP head: full-scale rerun (otoSpeech dev)

Rerun of the full-scale experiment on Modal, started from scratch: split regenerated (131 train / 16 dev / 20 gate / 253 excluded,
identical to the archived split), pilot caches rebuilt, 100-update VAP continuation retrained, frozen and continued
full caches (147 conversations each) rebuilt, then MLP and GRU heads trained with seeds 42 and 17 for 12 epochs each.
No TurnBench test or gate data was touched; dev is otoSpeech dev (16 conversations, 1031 positive events, 1930 negative spans).

**Provenance notes**
- The continued encoder is a fresh retraining (checkpoint SHA256 `584581008ef4c44d5e7f81ba675a0c73d1a926aa9942cff9cdfed7d483142918`), not the archived `2cd76b02...`; GPU nondeterminism means they differ.
- The VAP continuation ran on one A10G; the caches used batch size 8 (batch 16 ran out of memory in the STFT front end; `TD_CACHE_BATCH` makes it configurable).
- Several earlier detached Modal apps were cancelled externally and their runs resumed from epoch checkpoints; the head runs use the repo's `--resume` path. Final jobs were run as spawned calls on a deployed app (`modal_app.py` + `submit.py`).
- GRU: `temporal_head.py` (projection 128, GRU hidden 128, chunk 256 frames, lr 1e-3), run via `run_gru.py`. MLP: `heads.py` defaults via `run_full_scale.py`.

## Results

Operating point per run: max recall subject to FPR <= 0.10 on dev (the `heads.selection_key` rule), thresholds in 0.02 steps x recommit {none, 1.5 s}.
"final" = epoch 12; "best-epoch" = best over all 12 epochs (what `best.json` records; selected on dev, so optimistic).

| encoder | head | seed | epochs | final: recall / FPR / p50 ms | best-epoch (e): recall / FPR / p50 ms |
|---|---|---|---|---|---|
| continued | gru | 17 | 12 | 0.9457 / 0.0948 / 656 | 0.9544 / 0.0995 / 632 (e11) |
| continued | gru | 42 | 12 | 0.9515 / 0.0948 / 632 | 0.9544 / 0.0953 / 639 (e10) |
| continued | gru | **mean** | | 0.9486 / 0.0948 / 644 | 0.9544 / 0.0974 / 635 |
| continued | mlp | 17 | 12 | 0.9515 / 0.0964 / 632 | 0.9612 / 0.0979 / 596 (e4) |
| continued | mlp | 42 | 12 | 0.9476 / 0.1000 / 636 | 0.9573 / 0.0995 / 636 (e1) |
| continued | mlp | **mean** | | 0.9496 / 0.0982 / 634 | 0.9593 / 0.0987 / 616 |
| frozen | gru | 17 | 12 | 0.9525 / 0.0979 / 613 | 0.9554 / 0.0984 / 644 (e4) |
| frozen | gru | 42 | 12 | 0.9370 / 0.0953 / 628 | 0.9534 / 0.0959 / 652 (e5) |
| frozen | gru | **mean** | | 0.9447 / 0.0966 / 620 | 0.9544 / 0.0972 / 648 |
| frozen | mlp | 17 | 12 | 0.9496 / 0.0974 / 584 | 0.9554 / 0.0974 / 620 (e1) |
| frozen | mlp | 42 | 12 | 0.9253 / 0.0933 / 646 | 0.9573 / 0.1000 / 600 (e8) |
| frozen | mlp | **mean** | | 0.9374 / 0.0953 / 615 | 0.9564 / 0.0987 / 610 |

## Reading

- **GRU vs MLP: no measurable difference.** Final-epoch mean recall 0.945 (GRU) vs 0.937 (MLP) on the frozen encoder and 0.949 vs 0.950 on the continued encoder; best-epoch means 0.954 vs 0.956 and 0.954 vs 0.959. All inside the seed-to-seed spread of a single arm (frozen MLP final recall: 0.925 vs 0.950 across seeds). Median latency is, if anything, slightly worse for the GRU (about +5 to +40 ms).
- **Continued vs frozen encoder: also within noise** (final-epoch mean recall, continued vs frozen: GRU 0.949 vs 0.945, MLP 0.950 vs 0.937; best-epoch: GRU 0.954 vs 0.954, MLP 0.959 vs 0.956), consistent with the archived v1 gate result (continued did not beat frozen). The 100-update VAP continuation on 16 conversations is tiny; this says nothing about VAP at scale.
- **Statistical power is low.** With ~1031 positive events the binomial standard error of recall is about 0.7 points per run, and the best epoch is scattered over epochs 1-11, i.e. selecting it on the same 16 dev conversations mostly fits noise. Treat differences under ~1.5 points as unresolved. No confidence intervals or paired tests were computed.
- **Latency** (median matched ~600-650 ms here) is not comparable with the 375 ms reported on TurnBench dev for the archived policy: different data and a different, dev-selected threshold/recommit policy.
- **Conclusion:** a causal recurrent head over the existing per-frame features does not help. The hypothesis that the head was missing end-of-speech context is not supported by this experiment, so head capacity/memory is probably not the bottleneck.

## Suggested next steps (not run)
1. Spend effort on what changes the inputs or labels instead of the head: true segment starts for resume detection, soft/smoothed targets, the three-detector VAD gate, interruption head.
2. Use more data: train on all 420 conversations and select on public TurnBench dev (protocol amendment allows dev for development); this also gives a larger selection set than 16 conversations.
3. If VAP is revisited, scale it (all 131+ conversations, thousands of updates, more layers) before judging it.
4. Report per-conversation bootstrap intervals before any further architecture claim.
