# Complete Code Guide — Every File Explained

This explains **all 31 Python files** in `src/`, grouped by role. For each file:
what it does, its key functions, and how the code works. Read top-to-bottom and
you understand the whole project.

Legend: 🟢 setup · 🔵 data · 🟣 models · 🟠 training/eval · 🔴 research phases · ⚪ helpers

---

# GROUP 1 — Setup & data acquisition

## 🟢 `config.py` (137 lines) — the settings sheet
The single source of truth for every choice. No logic, only constants.
- **Paths:** where raw data, processed data, models, figures, metrics live.
- **`SENSOR_FEATURES`** = the 6 input sensors; **`TARGET` = "pH"**.
- **`LOOKBACK = 10`** (use last 10 minutes), **`HORIZON = 1`** (predict next minute).
- **Split ratios:** `TRAIN_RATIO=0.8`, `VAL_RATIO=0.1`.
- **Training:** `EPOCHS=30`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `SEEDS=[42,43,44]`.
- **Model sizes:** GRU 50 units, LSTM 43, Transformer d_model 64.
- **`VALID_RANGES`:** physically-possible min/max per sensor (used to drop junk).
- **`PAPER_TARGETS`:** the paper's reported numbers, for comparison.
Every other file imports from here, so changing one number changes it everywhere.

## 🟢 `download_data.py` (145 lines) — get the dataset from Kaggle
- `extract_local_zips()` — if you dropped a Kaggle ZIP in `data/raw/`, unzip it.
- `try_kaggle_api(ref)` — authenticate with your Kaggle token and download; if
  no exact name is known it **searches** Kaggle for the dataset slug.
- `main()` — tries: (1) local ZIP, (2) already-downloaded CSVs, (3) Kaggle API,
  and if all fail, prints manual instructions. This is why it "just worked."

## 🔵 `inspect_data.py` (22 lines) — peek at the raw files
Loops over every CSV in `data/raw/`, prints row counts, column names, pH range,
and the actuator column types. This is how we discovered the real schema and
that there were 5 different CSV files.

---

# GROUP 2 — Cleaning & shaping the data

## 🔵 `preprocess.py` (345 lines) — the kitchen prep (biggest data file)
Turns messy raw logs into a clean, model-ready table.
- **`CANONICAL_MAP` + `normalize_columns()`** — rename any header spelling to
  standard names (e.g. "ph", "pH_value" → `pH`).
- **`binarize_actuators()`** — convert "ON"/"OFF" text to 1.0/0.0 numbers.
- **`load_raw()`** — pick the best source CSV, load, normalize, binarize.
- **`ensure_timestamp()`** — parse the time column, sort chronologically.
- **`interpolate_defaults()`** — rows flagged `isDefault==1` are sensor failures;
  set them to "missing" then fill by **linear interpolation** (draw a straight
  line between the good values before/after). This is the paper's Eq. 8.
- **`validate_ranges()`** — mark physically-impossible readings (humidity 2013%,
  negative nutrients) as missing and interpolate them too (paper's range check).
- **`resample_time()`** — convert the every-10-seconds stream to every-1-minute
  using **time-based interpolation** (paper's Eq. 9).
- **`run()`** — full pipeline → saves `data/processed/clean_1min.csv`.
- **`run_paper_subset()`** — processes the authors' own cleaned subset →
  `clean_paper_subset.csv` (the smoother data used for the headline numbers).

## 🔵 `dataset.py` (197 lines) — turn the table into training examples
- **`WindowedData`** (a dataclass) — a neat container holding X_train/y_train,
  X_val/y_val, X_test/y_test, and the scalers.
- **`load_clean(source)`** — load either the full month or the paper subset.
- **`select_features()`** — choose which columns are inputs.
- **`_make_windows()`** — the core trick: slide a 10-row window across the data;
  each window's inputs = 10 minutes of sensors, its answer = the next pH.
- **`build(...)`** — the main function. It:
  1. fits a **MinMax scaler** (squashes values to 0–1) **on training data only**
     (so the model never peeks at test statistics — no leakage),
  2. builds the windows,
  3. splits into train/val/test using one of three strategies:
     - `random` → shuffle windows (the paper's way; **leaky**),
     - `chrono` → strict time order,
     - `temporal` → strict time order **with a gap** so test windows never
       overlap training windows (**the honest, leakage-free way we added**).
- **`inverse_target()`** — turn scaled predictions back into real pH numbers.

## 🔵 `metrics.py` (75 lines) — the scoring functions
Defines every score so all experiments compute them identically:
- `rmse`, `mae`, `r2` — the main three.
- `mape` — error as a percentage.
- `medae`, `maxae` — median and worst-case error.
- `within_tolerance(tol)` — % of predictions within ±tol pH (e.g. ±0.05).
- `all_metrics()` — runs all of them at once and returns a dictionary.

## 🔵 `feature_analysis.py` (78 lines) — which sensors matter?
Computes the **Pearson correlation** between every sensor and pH, draws the
heatmap (reproduces the paper's Figure 9), and prints which features are most
related to pH. Confirmed the exhaust fan barely matters, etc.

## ⚪ `diagnose_gap.py` (51 lines) — why is our error higher than the paper's?
A detective script. Measures the **persistence RMSE** (how much pH naturally
jumps step to step) for the full month vs the paper's subset. Showed the paper
used a *smoother* slice of data, which is why their error looked smaller — it was
about the data slice, not a bug.

---

# GROUP 3 — The models

## 🟣 `models.py` (163 lines) — the paper's 3 neural networks
- **`GRUModel`** — 2 stacked GRU layers (50 units) → dropout → 1 output. The
  paper's main model. We confirmed it has exactly 24,351 parameters.
- **`LSTMModel`** — 2 LSTM layers (43 units) → output. A GRU cousin.
- **`TransformerModel`** — input projection → positional encoding → encoder +
  decoder (attention) → output. The attention-based comparison model.
- **`PositionalEncoding`** — gives the Transformer a sense of time order.
- **`build_model(name, n_features)`** — factory: give it "gru"/"lstm"/
  "transformer" and it returns the model.
- **`count_parameters()`** — counts the model's tunable knobs.

## 🟣 `improved_gru.py` (148 lines) — OUR upgraded model
- **`GRUConfig`** — settings for the improved model (hidden size, layers,
  dropout, and the key switch **`delta_head`**).
- **`ImprovedGRU`** — the new architecture:
  - input **LayerNorm** (stabilizes scale),
  - GRU layers,
  - an MLP head,
  - the **delta head**: instead of outputting absolute pH, it outputs the
    *change* and adds it to the last observed pH. This is the single change that
    fixed the model (stops it from cheating by copying the last value).
- **`TrainConfig`** — training settings (AdamW optimizer, weight decay, Huber
  loss, LR scheduler, early stopping, gradient clipping).
- **`train_improved()`** — the robust training loop with early stopping (keeps
  the best model on validation, stops when it stops improving).

---

# GROUP 4 — Training, evaluation, reporting (the reproduction)

## 🟠 `train.py` (266 lines) — the training engine
- **`set_seed()`** — makes runs reproducible.
- **`make_loaders()`** — wraps data into PyTorch batches.
- **`train_one()`** — trains one model for 30 epochs, tracks loss, evaluates on
  test, measures latency and parameter count.
- **`train_model()`** — trains 3 times (3 seeds), reports mean ± std, saves the
  best model (`.pt`) and its predictions (`.npz`).
- **`main()`** — command-line entry: choose models, feature set, data source,
  split. Saves a results JSON.

## 🟠 `evaluate.py` (172 lines) — charts for the reproduction
Loads saved predictions and draws the paper's figures: actual-vs-predicted
(Fig 16), residual histograms (Fig 18), the ±0.05 tolerance plot (Fig 19), and
prints a comparison table vs the paper.

## 🟠 `baselines.py` (139 lines) — classical baselines (first round)
Runs **ARIMA** (grid-searches the best p,d,q order by AIC) plus **MLP, KNN,
XGBoost** and reports their scores — the paper's Table 6 comparison.

## 🟠 `make_report.py` (111 lines) — auto-write RESULTS.md
Reads the saved metric JSONs and generates `results/RESULTS.md`, a tidy summary
table of the reproduction. No manual typing of numbers.

## 🟠 `run_all.py` (60 lines) — one-click reproduction
Runs the whole original pipeline in order: download → preprocess (×2) →
feature analysis → train → evaluate → baselines → report.

## ⚪ `demo_predict.py` (58 lines) — quick live demo
Loads the saved GRU, runs it on test windows, prints actual-vs-predicted pH for
a few examples and the metrics. A tiny standalone demo.

## ⚪ `show_json.py` (15 lines) — pretty-print a results JSON
Turns an ugly JSON file into a readable table in the terminal. A convenience
tool (avoids Windows terminal quoting headaches).

---

# GROUP 5 — The research upgrade (the phase files)

## 🔴 `phase2_baseline.py` (128 lines) — honest baseline + persistence
Trains the GRU with fixed seeds and reports the **full metric set**. Crucially
adds the **persistence baseline** ("predict last value"). This is where we
discovered the GRU barely beats — and sometimes loses to — the trivial predictor.

## 🔴 `phase3_error_analysis.py` (135 lines) — where do errors come from?
Trains the GRU, finds the **20 largest errors**, and joins each to its data
context (was there a big pH jump? an actuator firing?). Proved that a couple of
glitchy spikes cause ~half the total squared error → explains why RMSE > MAE.

## 🔴 `phase4_baselines.py` (168 lines) — fair fight, all models
Runs **7 models** (Persistence, LinearRegression, RandomForest, ExtraTrees,
XGBoost, LSTM, GRU) under the **same protocol**, on both the leaky and honest
splits. This is the definitive comparison table. Showed the original GRU is the
worst on the honest split; ExtraTrees is the only clear winner over persistence.

## 🔴 `phase5_improve.py` (129 lines) — build the better GRU
Searches **32 configurations** of the ImprovedGRU, choosing the best **using the
validation set only** (test untouched until the very end). The winner is then
tested once. This is where the honest R² jumped from 0.03 → 0.65.

## 🔴 `confidence.py` (101 lines) — the "knows when it's unsure" engine
- **`mc_dropout_predict()`** — runs the model K times with dropout ON; the spread
  (std) of predictions = the model's **uncertainty**.
- **`simulate_switching()`** — the hybrid policy: trust the virtual sensor when
  confident; call the physical probe when uncertain or drifting; with hysteresis
  and a safety cadence so it doesn't flip-flop.

## 🔴 `phase7_confidence.py` (109 lines) — measure the switching
Loads the ImprovedGRU, computes uncertainty, calibrates thresholds on validation,
runs the switching simulation on the test stream, and reports the **measured
75.6% reduction** in physical-probe reads while keeping accuracy.

## 🔴 `phase8_failure_sim.py` (168 lines) — break the sensor on purpose
Injects realistic faults (missing / frozen / noisy / drifting / miscalibrated
pH) into 30% of the stream and compares a **traditional** system (trusts the
broken probe) vs the **proposed** system (falls back to the virtual sensor).
Traditional collapses (R²=−6.9); proposed stays accurate (R²=0.89).

## 🔴 `phase9_plots.py` (201 lines) — publication charts
Draws the upgraded-system figures: actual-vs-predicted, error-over-time,
residual histogram, ±0.05 tolerance, model-comparison bars, largest errors,
failure simulation, sensor usage, and uncertainty-vs-error.

## 🔴 `phase9_losscurve.py` (72 lines) — training curve
Retrains the winning ImprovedGRU config once while recording train/validation
loss each epoch, and plots the curve (shows healthy learning + early stopping).

## 🔴 `phase10_efficiency.py` (161 lines) — is it edge-ready?
Measures parameters, model size, CPU latency (mean & P95), throughput, and
memory. Then **exports to ONNX** (a portable format for tiny devices) and
verifies the ONNX output exactly matches PyTorch (parity = 0.0).

## 🔴 `phase12_export_demo.py` (84 lines) — prepare the dashboard data
Combines the Phase 7 and Phase 8 streams into one compact `demo_data.json` that
the dashboard animates (pH, confidence, mode, fault flags per time step).

## ⚪ `launch_dashboard.py` (44 lines) — run the live demo
Starts a tiny local web server over the `results/` folder and opens
`dashboard.html` in your browser (needed because browsers block reading local
files directly).

## 🔴 `run_all_phases.py` (53 lines) — one-click research upgrade
Runs Phases 2 → 12 in order with a single command.

## ⚪ `make_report_images.py` (244 lines) — the report as images
Reads the metric JSONs and renders clean PNG tables + a summary card into
`results/report_images/` (the images you just viewed).

---

# HOW THE FILES CONNECT (the flow)

```
download_data.py  →  data/raw/*.csv
      ↓
preprocess.py     →  data/processed/clean_1min.csv, clean_paper_subset.csv
      ↓
dataset.py  (uses config.py, called by everything below)
      ↓
   ┌────────────────────────────┬───────────────────────────────┐
   ↓                            ↓                               ↓
train.py (models.py)     phase4_baselines.py            phase5_improve.py
   ↓                            ↓                        (improved_gru.py)
evaluate.py                phase2/3 analysis                    ↓
   ↓                                                     phase7_confidence.py
make_report.py                                           (confidence.py)
                                                                ↓
                                                     phase8_failure_sim.py
                                                                ↓
                                        phase9_plots.py / phase10_efficiency.py
                                                                ↓
                                        phase12_export_demo.py → dashboard.html
```

Everything reads settings from `config.py` and scores from `metrics.py`, so the
whole system stays consistent.
```
