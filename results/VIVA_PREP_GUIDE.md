# Viva / Review Preparation Guide — Deep-Learning Concepts + File-by-File

This guide prepares you to answer **any** question your deep-learning faculty may
ask: every term, every design choice, and a walkthrough of each code file.
Read Part 1 (concepts) to speak fluently, then Part 2 (files) to defend the code,
then Part 3 (likely questions) to rehearse.

---

# PART 1 — DEEP-LEARNING CONCEPTS (explained in depth)

## 1.1 What kind of problem is this?
- **Supervised learning** — we have inputs (sensor readings) and known answers
  (the true pH), and the model learns the mapping.
- **Regression**, not classification — the output is a continuous number (pH
  5.3, 6.1, ...), not a category. That's why we use RMSE/MAE/R², not accuracy/
  precision/recall.
- **Time-series forecasting** — inputs arrive in time order and we predict a
  future value (next-minute pH) from a window of past values.

## 1.2 Features, target, window
- **Feature** = an input variable. Ours: pH, TDS (nutrient concentration),
  water level, DHT air temperature, DHT humidity, water temperature (6 sensors),
  plus 5 actuator signals for CARD.
- **Target (label)** = what we predict: next-step pH.
- **Look-back window (L=10)** = the model sees the last 10 one-minute readings
  of all features to predict step 11's pH. Shape of one sample: (10 timesteps ×
  n_features).
- **Horizon (H=1)** = we forecast one step (one minute) ahead.

## 1.3 Normalization (MinMax scaling)
- Neural nets train best when inputs are on a similar, small scale.
- **MinMax** maps each feature to [0,1]: `x' = (x - min) / (max - min)`.
- **Critical detail (no leakage):** we fit the scaler on the **training data
  only**, then apply those same min/max to val and test. Fitting on all data
  would leak test information into training. (See dataset.py.)
- At the end we **inverse-transform** predictions back to real pH units for
  reporting.

## 1.4 Train / Validation / Test split
- **Train** — the model learns its weights from this.
- **Validation** — used to choose settings (which model/epoch is best) WITHOUT
  touching test. Prevents us from "tuning on the test set."
- **Test** — touched only once at the very end; the honest estimate of
  generalization.
- **Random split vs Temporal split** (the project's big theme):
  - *Random*: shuffle windows into train/test. Because neighbouring windows
    overlap by 9 of 10 timesteps, test windows are nearly identical to training
    ones → **temporal data leakage** → inflated scores (R²≈0.98 even for a
    trivial baseline).
  - *Temporal*: train on the earlier period, test on a later/future period,
    with a **gap (embargo)** so no test window shares any row with training.
    This is the **honest** protocol.

## 1.5 Neural network building blocks
- **Neuron / weight / bias** — each connection has a weight; a neuron computes
  `w·x + b` then a nonlinear activation.
- **Activation functions**:
  - **Sigmoid** σ(x)=1/(1+e^⁻ˣ) → squashes to (0,1); used for gates.
  - **tanh** → squashes to (−1,1); used inside GRU/LSTM for candidate states.
  - **ReLU** max(0,x) → cheap nonlinearity in our MLP heads.
- **Fully-connected / Dense / Linear layer** — `y = Wx + b`; our output heads.
- **Dropout** — randomly zeroes a fraction of activations during training so the
  net can't over-rely on any unit → reduces **overfitting**. (Also reused at
  inference for MC-Dropout uncertainty.)
- **LayerNorm** — normalizes a layer's inputs to zero-mean/unit-variance per
  sample; stabilizes and speeds up training (we apply it to the input window).

## 1.6 RNNs, LSTM, GRU (the core of the project)
- **RNN (Recurrent Neural Network)** — processes a sequence step by step,
  carrying a **hidden state** h_t that summarizes the past. Problem: vanilla RNNs
  suffer **vanishing/exploding gradients** on long sequences.
- **LSTM (Long Short-Term Memory)** — adds a **cell state** and three gates
  (input, forget, output) that control what to remember/forget → handles long
  dependencies. More parameters.
- **GRU (Gated Recurrent Unit)** — a simpler LSTM with **two gates**:
  - **Reset gate r** — how much past state to forget when forming the candidate.
  - **Update gate z** — how much to blend old state vs new candidate:
    `h_t = (1−z)·h̃ + z·h_{t−1}`.
  - Fewer parameters than LSTM, similar accuracy → faster, better for edge
    devices. This is why the paper and we prefer GRU.
- **Stacked layers** — we use 2 GRU layers; the second reads the first's output
  sequence, learning higher-level temporal patterns.
- **Transformer** — uses **self-attention** (every timestep attends to all
  others) instead of recurrence. Powerful but here ~14× more parameters for no
  accuracy gain, so not chosen.

## 1.7 Training mechanics
- **Loss function** — the number the model minimizes:
  - **MSE** (mean squared error) — standard for regression; squares errors so
    large errors dominate.
  - **Huber loss** — quadratic for small errors, linear for large ones → robust
    to outlier spikes (a few sensor glitches don't dominate).
  - **Tolerance-Weighted Huber (our novel loss)** — zero cost inside ±τ
    (errors too small to matter agronomically), Huber outside. Optimizes the
    "within-tolerance" objective directly.
- **Optimizer** — the rule that updates weights from gradients:
  - **SGD** — basic gradient descent.
  - **Momentum** — adds a velocity term so updates accumulate in consistent
    directions, smoothing and speeding convergence (think of a ball rolling
    downhill). *If asked "did you use momentum?":* yes, implicitly — Adam/AdamW
    maintain a running average of past gradients (the "first moment," β₁≈0.9),
    which is exactly momentum.
  - **Adam** — Adaptive Moment Estimation: combines momentum (1st moment) with a
    per-parameter adaptive learning rate from the 2nd moment (squared
    gradients). Robust default.
  - **AdamW** — Adam with **decoupled weight decay** (see below). We use AdamW
    for the Improved GRU and CARD.
- **Learning rate (lr=0.001)** — step size for weight updates. Too big →
  diverge; too small → slow.
- **Learning-rate scheduler (ReduceLROnPlateau)** — automatically halves the lr
  when validation stops improving, for finer convergence.
- **Weight decay (1e-4) / L2 regularization** — penalizes large weights to
  reduce overfitting. "Decoupled" in AdamW means it's applied directly to
  weights, not through the gradient (cleaner than classic L2 in Adam).
- **Batch size (32)** — number of samples per gradient update. **Epoch** — one
  full pass over the training set (we train up to 30–120).
- **Backpropagation** — the algorithm that computes gradients of the loss w.r.t.
  every weight via the chain rule; the optimizer then updates weights.
- **Gradient clipping (max-norm 1.0)** — caps the gradient size to prevent
  exploding-gradient instability (common in RNNs).
- **Early stopping** — stop training when validation loss stops improving and
  keep the **best** checkpoint, not the last → avoids overfitting.
- **Random seed (42/43/44)** — fixes the random number generator so runs are
  reproducible; we train 3 seeds and report mean±std.

## 1.8 Evaluation metrics (define each precisely)
- **MAE** = mean(|y−ŷ|) — average error in pH units. Robust to outliers.
- **RMSE** = sqrt(mean((y−ŷ)²)) — penalizes big errors; ≥ MAE always.
- **R²** = 1 − SS_res/SS_tot — fraction of variance explained (1=perfect, 0=as
  good as predicting the mean, <0=worse than the mean). **Not an accuracy %.**
- **MAPE** — mean absolute percentage error.
- **MedAE / MaxAE** — median / maximum absolute error (robustness + worst case).
- **within ±0.05 pH** — % of predictions within a 0.05 pH tolerance band; the
  agronomically meaningful "is it good enough" metric (not classification
  accuracy).

## 1.9 Overfitting / underfitting / generalization
- **Overfitting** — model memorizes training data, fails on new data (big train-
  test gap). Mitigated by dropout, weight decay, early stopping.
- **Underfitting** — model too simple to capture the pattern (high train error).
- **Generalization** — performance on unseen data; the temporal split measures
  it honestly.
- **Persistence baseline** — "predict next = last value." The sanity check every
  model must beat; on slow-moving pH it's near the achievable ceiling.

## 1.10 Uncertainty & the novel parts
- **MC-Dropout** — keep dropout ON at inference, run K forward passes; the
  spread (std) of predictions estimates **epistemic uncertainty** (model
  confidence). Cheap Bayesian approximation.
- **Heteroscedastic uncertainty (CARD's σ head)** — the model outputs a
  per-step variance; trained with **Gaussian NLL** so the predicted σ tracks the
  real error. "Heteroscedastic" = uncertainty varies per input.
- **Gaussian NLL** — negative log-likelihood of a Gaussian; minimizing it makes
  σ large where the model is error-prone and small where it's confident.
- **CARD's dosing gate** — a sigmoid gate g∈[0,1] from recent actuator activity;
  g≈0 when the system is quiet (anchor to persistence), g≈1 during dosing
  (allow deviation). Supervised by a light **BCE-style gate prior**.
- **CASC controller** — an **integral control law** (not learning) that nudges
  the uncertainty threshold each step to hit a target physical-probe usage
  budget: `thr ← thr·(1 + kp·(rate − budget))`.

---

# PART 2 — FILE-BY-FILE WALKTHROUGH

> Run convention: all scripts run from `src/` using `..\.venv\Scripts\python.exe`.
> Settings live in `config.py`; metrics in `metrics.py`.

## config.py — central settings
Holds every constant: paths, feature lists (`SENSOR_FEATURES`,
`ACTUATOR_FEATURES`), `LOOKBACK=10`, `HORIZON=1`, split ratios
(`TRAIN_RATIO=0.8`, `VAL_RATIO=0.1`), training hyperparameters
(`EPOCHS=30`, `BATCH_SIZE=32`, `LEARNING_RATE=1e-3`, `SEEDS=[42,43,44]`), model
sizes, `VALID_RANGES` (physical plausibility bounds), `RESIDUAL_TOLERANCE=0.05`,
and the paper's reported targets for comparison. *Why:* one source of truth →
reproducible, no magic numbers scattered in code.

## download_data.py — data acquisition
Downloads the Kaggle dataset (via API token or a manual ZIP fallback) into
`data/raw/`. *Talking point:* dataset DOI 10.34740/kaggle/ds/8223904, CC BY 4.0.

## preprocess.py — cleaning pipeline
1. `normalize_columns` — standardize varied header names.
2. `binarize_actuators` — ON/OFF → 1/0.
3. `interpolate_defaults` — rows flagged `isDefault==1` (sensor failures) are set
   to NaN and filled by **linear interpolation** (paper Eq. 8).
4. `validate_ranges` — physically impossible readings (e.g. humidity>100%, TDS<0)
   are treated as faults and interpolated (paper's range check).
5. `resample_time` — convert 10-second cadence → 1-minute via **time-based
   interpolation** (paper Eq. 9).
Outputs `clean_1min.csv` (full month) and `clean_paper_subset.csv` (the authors'
smoother subset used for headline numbers). *Why interpolation:* keeps the time
axis continuous without deleting rows.

## dataset.py — windowing, scaling, splitting
- `WindowedData` dataclass bundles X/y for train/val/test + the scalers.
- `_make_windows` — slides the L=10 window; X=(N,10,features), y=next pH.
- `build(...)` — fits MinMax **on the first 80% only** (no leakage), transforms
  all, then splits by `split=`:
  - `random` (leaky, reproduces paper),
  - `chrono` (time order),
  - `temporal` (time order + `gap=LOOKBACK` embargo → the honest one).
- `inverse_target` — scaled pH → physical pH.
*Key viva point:* explain the embargo gap and why it removes leakage.

## metrics.py — evaluation functions
`rmse, mae, r2, mape, medae, maxae, within_tolerance(tol)`, and `all_metrics`
that returns them all including within ±0.01/0.03/0.05/0.10 pH. *Why multiple
tolerance bands:* shows the full error distribution, not one cherry-picked
number.

## models.py — the paper's models
- `GRUModel` — 2× GRU(hidden) → dropout → Dense(1). Reproduces the paper's exact
  24,351 parameters (a strong correctness check).
- `LSTMModel` — 2× LSTM(43) → Dense(1).
- `TransformerModel` — input projection → **positional encoding** → encoder+
  decoder (multi-head self-attention) → Dense(1).
- `PositionalEncoding` — injects timestep order into the Transformer (which has
  no inherent notion of order).
- `count_parameters` — sums trainable weights.

## improved_gru.py — our improved model + robust trainer
- **Delta head**: predicts the *change* in pH and adds it to the last value
  (`pH_hat = last_pH + δ`), so the net learns the deviation from persistence —
  the fix that makes it generalize on the honest split.
- Input **LayerNorm**, MLP head, dropout.
- `train_improved` — **AdamW + weight decay + Huber loss + ReduceLROnPlateau +
  gradient clipping + early stopping** (best-on-validation restore).

## train.py — training engine for the paper models
- `set_seed` for reproducibility; `make_loaders` builds batches.
- `train_one` — 30-epoch loop (Adam, lr 1e-3, MSE), tracks loss, measures
  inference **latency** and parameter count.
- `train_model` — 3 seeds, reports mean±std, saves best `.pt` + predictions.

## confidence.py — Phase 7 reliability engine
- `mc_dropout_predict` — K stochastic passes → mean + std (uncertainty).
- `simulate_switching` — hysteresis policy: trust virtual pH when confident,
  read the physical probe when uncertain or drifting (EMA of error), with a
  safety cadence. Measures the reduction in physical reads.

## card.py — the NOVEL algorithm (study this closely)
- `CARD` model: LayerNorm → stacked GRU → three heads:
  - **delta_head** → the pH change δ,
  - **gate_head** (sigmoid, fed GRU state + recent actuator vector) → g∈[0,1],
  - **logvar_head** → log-variance (uncertainty).
  Output: `pH_hat = last_pH + g·δ` (physics-anchored adaptive skip).
- `tolerance_weighted_huber` — the novel loss (dead-zone + Huber).
- `gaussian_nll` — trains the σ head (heteroscedastic).
- `gate_dosing_prior` — BCE-style term pushing g→1 during dosing, g→0 when quiet
  (makes the gate interpretable; validated corr≈+0.98).
- `train_card` — AdamW + scheduler + grad clip + early stopping on the TWH val
  loss.
*Scientific insight to mention:* τ (dead-zone) must be **smaller than the typical
error**, else ~99% of samples get zero gradient and the model can't learn — we
set τ=0.003 (persistence error ≈0.002 scaled).

## card_eval.py — CASC controller + evaluation
- `casc_controller` — integral control law adjusting the uncertainty threshold
  to hit a target probe-usage budget; measured to hit 5/10/20% within ~0.2%.
- Compares CARD vs the baselines and saves `card_results.json`.

## card_ablation.py — what each component contributes
Trains 5 variants (full, no-gate, no-TWH, no-uncertainty, no-gate-prior) and
reports each one's effect — honest evidence of which pieces help.

## Reporting / dashboard scripts
- `evaluate.py`, `phase9_plots.py`, `card_plots.py` — generate figures.
- `phase10_efficiency.py` — params, size, CPU latency, **ONNX** export (portable
  edge format) with exact parity check.
- `b7_tolerance_bands.py`, `d11_cost_calculator.py`, `d12_pdf_report.py` —
  tolerance breakdown, cost/lifespan estimate, PDF summary.
- `build_results_explorer.py` — builds the single-file interactive dashboard.

---

# PART 3 — LIKELY FACULTY QUESTIONS (with crisp answers)

**Q: Why GRU over LSTM/Transformer?**
A: GRU matches LSTM accuracy with fewer gates/parameters (faster, edge-friendly);
the Transformer was ~14× larger here for no accuracy gain.

**Q: Did you use momentum?**
A: Yes — Adam/AdamW's first-moment estimate (β₁≈0.9) is exactly momentum; it also
adds a per-parameter adaptive learning rate via the second moment.

**Q: What is weight decay and why AdamW not Adam?**
A: Weight decay is L2 regularization to curb overfitting. AdamW decouples it from
the gradient update, which is the mathematically correct form and more stable.

**Q: Why is R² not accuracy?**
A: This is regression. R² is the fraction of variance explained; "accuracy" is a
classification term. We report within-±0.05 pH as the intuitive quality metric.

**Q: Your within-±0.05 is 94% — is that impressive?**
A: Honestly, it's a within-tolerance rate, and a trivial persistence baseline
also hits ~94% because pH moves slowly. So that number isn't our contribution;
our value is the 76% sensor-usage reduction, fault robustness (R²=0.89 vs −6.9),
and budget control.

**Q: What is data leakage and how did you fix it?**
A: In the random split, overlapping windows let test data resemble training data,
inflating R² to 0.98 (even persistence hits it). We built a temporal split with a
LOOKBACK-sized embargo gap so no test window overlaps training → honest R².

**Q: How do you prevent overfitting?**
A: Dropout, weight decay, early stopping on validation, gradient clipping, and a
small model (10,733 params).

**Q: What is the vanishing gradient problem and how do GRUs help?**
A: In deep/long RNNs gradients shrink exponentially, stalling learning. GRU's
update/reset gates create near-identity paths for the hidden state, letting
gradients flow → long-range dependencies are learnable.

**Q: Why Huber / Tolerance-Weighted Huber?**
A: MSE is dominated by a few sensor-spike outliers (we showed 1 point = 49% of
SSE). Huber is robust to them; our TWH additionally ignores sub-tolerance errors
to optimize the metric that matters.

**Q: How does MC-Dropout give uncertainty?**
A: Dropout left on at inference makes each forward pass a sample from an
approximate posterior; the variance across K passes is epistemic uncertainty.

**Q: Is CARD actually better?**
A: On raw accuracy it ties the baselines (near-ceiling data). Its contribution is
system-level: interpretable dosing gate (corr≈+0.98), calibrated uncertainty,
and a controller that hits a chosen sensor-usage budget within ~0.2%. We state
this honestly.

**Q: What would you do with more time?**
A: Multi-step forecasting (where smart models should beat persistence),
multi-seed significance tests, cross-cycle/crop data, and a real (not simulated)
fault + probe-wear study.

---

*Tip:* if you can explain §1.6 (GRU gates), §1.4 (the leakage/temporal split),
and §1.7 (optimizer/loss/regularization) in your own words, you can handle most
of the viva. Keep the honest framing — faculty reward rigor over inflated claims.
