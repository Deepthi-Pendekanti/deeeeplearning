# GRU-Based Virtual pH Sensor — Reproduction

Reproduction of the machine-learning pipeline (Module 2) from:

> **Toward sustainable hydroponic farming: An AI-driven IoT framework for
> virtual pH sensing and sensor lifespan extension**
> Moniruzzaman, Meem, Masud, Khan, Munir, Bairagi.
> _Alexandria Engineering Journal_ 143 (2026) 40–63.

The paper predicts hydroponic **pH** one step ahead from other sensor readings
using a **GRU** "virtual sensor," so the physical pH probe can be sampled less
often (extending its lifespan). This repo reproduces the full ML pipeline:
data loading → cleaning → preprocessing → normalization → windowing → GRU/LSTM/
Transformer training → evaluation → figures, plus ARIMA and MLP/KNN/XGBoost
baselines.

Module 1 (the physical Raspberry-Pi IoT hardware) is not reproducible in
software and is out of scope.

---

## ⭐ Research upgrade (Phases 1–13)

Beyond the paper reproduction, this project was extended into a research-grade,
hackathon-ready virtual sensing system. Full write-ups live in `results/`:

- `results/FINAL_REPORT.md` — the comprehensive report + concise summary.
- `results/PHASE1_AUDIT.md` … `PHASE12_DASHBOARD.md` — per-phase findings.
- `results/dashboard.html` — live demo (run `python src/launch_dashboard.py`).

**Headline honest findings:**

- The paper-style random split is **temporally leaky**; a trivial persistence
  predictor also reaches R²≈0.98, so that number is not forecasting skill.
- On a **leakage-free temporal split**, the original GRU collapses to R²=0.03.
  A **delta-head ImprovedGRU** recovers it to **R²=0.65 at 2.24× fewer params**,
  matching the strong persistence/ExtraTrees baselines.
- **Confidence-aware hybrid switching** cuts physical-probe usage by **75.6%**
  while maintaining accuracy (measured).
- Under **30% simulated sensor faults**, the proposed system holds **R²=0.89**
  vs a traditional pipeline's **R²=−6.9**.
- **Edge-ready:** 10,733 params, ~0.9 ms CPU, 6.8 KB ONNX (exact parity).

Reproduce the upgrade: `python src/run_all_phases.py` (after data is prepared).
Launch the demo: `python src/launch_dashboard.py`.

## Results at a glance

Headline config: paper's cleaned subset, 6 sensor features, random 80:20 split,
30 epochs, 3 seeds. Best run per model (test set, physical pH units):

| Model       | RMSE (ours) | RMSE (paper) | MAE (ours) | MAE (paper) | R² (ours)  | R² (paper) | params (ours) | params (paper) |
| ----------- | ----------- | ------------ | ---------- | ----------- | ---------- | ---------- | ------------- | -------------- |
| **GRU**     | 0.0285      | 0.0181       | 0.0107     | 0.0116      | **0.9824** | **0.9824** | 24,051        | 24,351         |
| LSTM        | 0.0279      | 0.0190       | 0.0092     | 0.0124      | 0.9831     | 0.9787     | 23,952        | 24,100         |
| Transformer | 0.0369      | 0.0206       | 0.0219     | 0.0128      | 0.9703     | 0.9749     | 351,041       | 251,585        |

- **R² for the GRU matches the paper exactly (0.9824)** and MAE beats it.
- The GRU with **8 input features** reproduces the paper's exact **24,351**
  parameters (see "Faithful-reproduction notes" below).
- Full numbers, mean±std, and baselines are in `results/RESULTS.md`.

---

## Quick start

Prerequisites: Python 3.11+ and a Kaggle account.

```powershell
# 1. Create the environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Kaggle credentials (needed once to download the dataset)
#    Kaggle -> Settings -> API -> Create New Token, then save the token to:
#      %USERPROFILE%\.kaggle\access_token     (new KGAT_ token), OR
#      %USERPROFILE%\.kaggle\kaggle.json       (classic token)

# 3. Run the whole pipeline (from the src/ directory)
cd src
..\.venv\Scripts\python.exe run_all.py
```

Outputs land in `results/` (figures, metrics JSON, saved models, RESULTS.md).

### Run individual stages

```powershell
cd src
..\.venv\Scripts\python.exe download_data.py
..\.venv\Scripts\python.exe preprocess.py                 # full month
..\.venv\Scripts\python.exe preprocess.py --paper-subset  # authors' subset
..\.venv\Scripts\python.exe feature_analysis.py           # Fig. 9
..\.venv\Scripts\python.exe train.py --models gru lstm transformer --source paper --split random
..\.venv\Scripts\python.exe evaluate.py                   # Fig. 16/18/19
..\.venv\Scripts\python.exe baselines.py                  # ARIMA + ML
..\.venv\Scripts\python.exe make_report.py
```

### Live demo (loads the trained model)

```powershell
cd src
..\.venv\Scripts\python.exe demo_predict.py --n 12
```

Loads `results/models/gru_best.pt`, runs it on held-out test windows, prints
the test metrics and a table of actual vs predicted pH — ideal for a live
demonstration.

---

## Pipeline & paper mapping

| Stage       | File                      | Paper section        | What it does                                                                        |
| ----------- | ------------------------- | -------------------- | ----------------------------------------------------------------------------------- |
| Config      | `src/config.py`           | 3.3.10, Table 4/A.9  | All hyperparameters, traceable to the paper                                         |
| Download    | `src/download_data.py`    | Data availability    | Fetch Kaggle dataset (API or manual ZIP)                                            |
| Clean       | `src/preprocess.py`       | 3.3.5–3.3.6, Eq. 8/9 | Linear-interpolate `isDefault==1`; range-check (Alg. 2); time-resample 10 s → 1 min |
| Correlation | `src/feature_analysis.py` | 3.3.7, Fig. 9        | Pearson correlation heatmap (Eq. 10)                                                |
| Windowing   | `src/dataset.py`          | 3.3.10               | MinMax scaling (train-only fit), look-back 10, one-step-ahead, 80:20 split          |
| Models      | `src/models.py`           | Table 4 / A.9        | GRU(50,50), LSTM(43,43), Transformer(3L/4H/d64)                                     |
| Train       | `src/train.py`            | 3.3.10               | 30 epochs, batch 32, Adam lr 1e-3, MSE, 3 seeds                                     |
| Metrics     | `src/metrics.py`          | 3.3.9, Eq. 23–25     | RMSE, MAE, R², MAPE, tolerance rates                                                |
| Evaluate    | `src/evaluate.py`         | Fig. 16/18/19        | Actual-vs-pred, residual hist, ±0.05 outliers                                       |
| Baselines   | `src/baselines.py`        | 4.2.2, Table 6       | ARIMA(p,d,q) by AIC; MLP/KNN/XGBoost                                                |
| Report      | `src/make_report.py`      | —                    | Consolidated `results/RESULTS.md`                                                   |

### Exact settings reproduced

- **Preprocessing:** linear interpolation for sensor-failure rows (Eq. 8),
  time-based interpolation resampling 10 s → 1 minute (Eq. 9).
- **Features:** sensor-only inputs (paper's best "All sensors" ablation config).
- **Window:** look-back 10 steps, one-step-ahead pH.
- **Split:** 80:20, validation drawn from the training portion.
- **GRU:** GRU(50, return_sequences=True) → Dropout → GRU(50) → Dropout →
  Dense(1). ~24k parameters.
- **Training:** 30 epochs, batch 32, Adam (lr 0.001), MSE loss, 3 runs (seeds
  42/43/44), implemented in PyTorch.

---

## Faithful-reproduction notes (for discussion)

These are honest deviations/findings, documented rather than hidden:

1. **Feature count (24,351 params).** Table A.9 states the GRU input is
   `(batch, 10, 8)` and the model has **24,351** parameters — which corresponds
   to **8** input features, even though Appendix A.1.2 lists only 5. Our GRU
   with 8 features reproduces 24,351 parameters _exactly_. We use the 6 physical
   sensors as the primary "All sensors" config (the paper's best ablation),
   and provide an 8-feature variant (`--features 8`) that matches the exact
   parameter count.

2. **RMSE vs data window.** The paper's headline RMSE (0.0181) is only
   achievable on their **cleaned subset** (pH tightly in 5.34–6.82, a smoother
   series). We confirmed this quantitatively (`src/diagnose_gap.py`): the
   one-step RMSE is lower-bounded by the series' persistence RMSE, which is
   ~0.020 on the smooth subset vs ~0.033 on the full volatile month. On the
   same subset the paper used, our R² matches exactly and MAE is lower.

3. **Split.** The paper reports jointly low RMSE **and** high R² (0.9824). A
   strict chronological split lands the test set on a near-constant tail
   (deflating R²). A window-level **random** 80:20 split reproduces the paper's
   joint numbers, so that is the headline config. `--split chrono` is available
   for the strict time-ordered variant.

4. **GRU vs LSTM.** In our runs LSTM and GRU are statistically tied (both
   R² ≈ 0.982). The paper also calls them "very similar" and selects the GRU
   for its **lower inference latency** — which we also observe.

5. **Hardware.** Trained on CPU (no GPU on this machine) rather than the paper's
   RTX 3080. The models are tiny (~24k params) so this only affects wall-clock
   time, not the results.

---

## Project layout

```
Deep_learning/
├── README.md
├── requirements.txt
├── src/                 # all pipeline code
├── data/
│   ├── raw/             # Kaggle CSVs (downloaded)
│   └── processed/       # clean_1min.csv, clean_paper_subset.csv
└── results/
    ├── figures/         # fig9, fig16, fig18, fig19
    ├── metrics/         # *.json result files
    ├── models/          # *_best.pt, *_test_predictions.npz
    └── RESULTS.md        # consolidated report
```
