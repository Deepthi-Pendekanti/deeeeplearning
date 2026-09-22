# Phase 5 — Improved GRU (validation-driven, leakage-free temporal split)

## Method
32-config grid searched on the **validation** set of the temporal (leakage-free)
split; the single best-by-val-MAE config was evaluated **once** on the test set.

Knobs searched: delta-head {T/F}, loss {Huber/MSE}, hidden {32,64},
layers {1,2}, dropout {0.1,0.3}. All configs used: input LayerNorm, AdamW,
weight decay 1e-4, ReduceLROnPlateau scheduler, gradient clipping 1.0, early
stopping (patience 12, best-on-val restore).

## The decisive factor: the delta head
Predicting the pH **change** (then adding back the last observed pH) instead of
absolute pH:

| delta_head | typical val R² |
|---|---|
| **True** | **0.71 – 0.72** (all 16 configs) |
| False | mostly negative to ~0.34 (unstable) |

Every absolute-pH config failed to generalize; every delta-head config
generalized. This is the principled cure for the persistence-dominance /
leakage problem: the network only learns the deviation from persistence.

## Best config
delta-head, Huber loss (δ=0.01), hidden=32, layers=2, dropout=0.1.
**Parameters: 10,733** (vs 24,051 baseline — 2.3× smaller).

## Test result (temporal, leakage-free) — HONEST

| Model | R² | MAE | RMSE | ±0.05 pH |
|---|---|---|---|---|
| GRU baseline (as-published-style) | 0.0304 | 0.0390 | 0.0437 | 72.28% |
| Persistence | 0.6500 | 0.0196 | 0.0263 | 93.78% |
| **Improved GRU** | **0.6499** | **0.0196** | **0.0263** | **94.43%** |

## Honest interpretation
- The improved GRU is a **massive fix** over the baseline (R² 0.03 → 0.65) and is
  **2.3× smaller**.
- It **matches persistence** on R²/MAE/RMSE and **slightly exceeds** it on the
  ±0.05 tolerance rate (94.43% vs 93.78%).
- We do **not** claim it substantially beats persistence: on a series this
  autocorrelated, persistence is a near-ceiling baseline. The GRU's added value
  is not raw accuracy but (a) robustness when the physical pH input is
  missing/faulty and (b) usable uncertainty — developed and measured in
  Phases 7–8.

Artifacts: `results/metrics/phase5_search.json`,
`results/models/improved_gru_best.pt`,
`results/models/improved_gru_test_predictions.npz`.
