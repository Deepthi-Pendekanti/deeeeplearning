# CARD — Faculty Explanation Guide (all CARD files)

CARD = **Confidence-Adaptive Residual Dosing-aware forecaster**, the novel
algorithm of this project. It is a *fusion* of four ideas designed together for
virtual pH sensing. This guide explains the three source files
(`card.py`, `card_eval.py`, `card_ablation.py`) at the depth a deep-learning
faculty will expect: the concept, the math, the code, and the likely questions.

---

## 0. The one-sentence summary
CARD predicts the next pH as **last pH + (dosing-gate × learned change)**,
trains with a **tolerance-weighted loss**, also predicts **its own uncertainty**,
and uses that uncertainty in a **controller** that decides when to read the
real probe — hitting a chosen sensor-usage budget.

The core prediction equation (memorize this):
```
pH_hat(t) = pH(t-1) + g(t) · δ(t)
```
- `pH(t-1)` = last observed pH (the persistence anchor)
- `δ(t)` = learned change (the "residual")
- `g(t) ∈ [0,1]` = dosing gate (0 = trust persistence, 1 = allow full deviation)

---

# FILE 1 — `card.py` (the model + loss + training)

This file defines the network, the novel loss, the uncertainty objective, and
the training loop. It has four conceptual components.

## Component 1 — Physics-anchored adaptive residual skip
**Problem it solves:** pH changes very slowly, so a model that predicts absolute
pH can "cheat" by copying the last value and still score well — it never learns
real structure and fails on future data. 
**Solution:** predict only the *change* δ, and gate it by how active the system
is. When no acid/nutrient is being dosed, `g→0`, so the prediction = persistence
(safe). When dosing occurs, `g→1`, so the model is free to predict a real change.

Code (in `CARD.forward`):
```python
last_ph   = x[:, -1, target_index]        # persistence anchor
recent_act = x[:, -1, n_sensor:]          # last actuator (dosing) vector
z = GRU(LayerNorm(x))[:, -1, :]           # final hidden state
delta = delta_head(z)                     # learned change
gate  = sigmoid(gate_head([z, recent_act]))   # g in [0,1]
pH_hat = last_ph + gate * delta           # the adaptive skip
```
- **`gate_head`** takes the GRU state **plus the raw recent actuator vector**, so
  the gate is explicitly dosing-aware.
- **sigmoid** forces `g ∈ [0,1]` (a soft 0/1 switch).

## Component 2 — Tolerance-Weighted Huber (TWH) loss
**Problem:** standard MSE wastes capacity chasing tiny errors that don't matter
agronomically (anything within ±0.05 pH is "fine"), and is dominated by a few
sensor-glitch spikes.
**Solution:** a loss that is **zero inside a ±τ dead-zone** and **Huber-shaped
outside**:
```python
err    = |pred - target|
excess = max(err - tau, 0)                 # ignore sub-tolerance error
loss   = 0.5*excess^2        if excess <= delta   # quadratic (small)
         delta*(excess-0.5*delta) otherwise       # linear (robust to spikes)
```
- **Dead-zone (τ):** errors below τ cost nothing → the model only works on
  errors that matter.
- **Huber part:** quadratic near zero, linear far out → robust to outliers
  (unlike MSE which squares them).

## Component 3 — Self-supervised uncertainty head
**Idea:** the model predicts its own error bar. A second head outputs a
**log-variance** `s`; the model is trained so σ = exp(s/2) is large where it's
likely wrong and small where it's confident. This is **heteroscedastic**
(input-dependent) uncertainty, trained with **Gaussian negative log-likelihood**:
```python
NLL = 0.5 * ( (pred-target)^2 / var  +  log(var) )     # var = exp(s)
```
Minimizing this balances two forces: the `1/var` term pushes σ down to fit well;
the `log(var)` term penalizes pretending to be certain. The result is a
*calibrated* uncertainty used later by the controller.

## Component (gate prior) — making the gate interpretable
A light **BCE-style penalty** nudges the gate toward 0.9 when dosing is active
and 0.1 when quiet, so `g` is physically meaningful (not just any value that
happens to minimize error):
```python
target_gate = 0.1 + 0.8 * (actuator_active)      # 0.9 if dosing else 0.1
gate_prior  = BCE(gate, target_gate)
```
Measured result: corr(gate, dosing activity) ≈ **+0.98** — it genuinely opens
during dosing.

## The training loop (`train_card`)
```
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)      # momentum + L2 regularization
scheduler = ReduceLROnPlateau(factor=0.5, patience=5)  # lower lr when stuck
loss = TWH(pH_hat, y) + 0.2*NLL + 0.3*gate_prior
backprop → clip gradients (max-norm 1.0) → optimizer step
early stopping (patience=20) on validation TWH; restore best weights
```
Everything standard here — AdamW, scheduler, gradient clipping, early stopping —
is textbook good practice; the novelty is the **composite loss** and the
**gated residual output**.

### Likely faculty questions on card.py
- *"Why predict the change instead of pH?"* → Removes the persistence shortcut;
  forces learning the deviation, which is the useful signal. Proven in ablation
  (variant B drops when gate forced to 1).
- *"Why a dead-zone in the loss?"* → To optimize the agronomic within-tolerance
  objective directly, and ignore sensor noise.
- *"Critical insight on τ?"* → τ must be **smaller than the typical error**
  (~0.002 scaled). At τ=0.02, ~99% of samples fell inside the dead-zone → zero
  gradient → the model couldn't learn. We use τ=0.003.
- *"How is the uncertainty trained/validated?"* → Gaussian NLL; validated by a
  positive correlation between predicted σ and actual error.
- *"Did you use momentum?"* → Yes, via AdamW (first-moment estimate β₁≈0.9) plus
  decoupled weight decay.

---

# FILE 2 — `card_eval.py` (Component 4 controller + evaluation)

This file trains CARD, evaluates it honestly, and implements the **CASC
controller** (Component 4).

## Component 4 — Confidence-Adaptive Sampling Controller (CASC)
**Goal:** reduce physical-probe reads to a chosen **budget** (e.g. "use the probe
≤10% of the time") while keeping accuracy. This is a **control-theory** idea, not
learning.

Logic (`casc_controller`):
```
threshold thr = quantile of sigma for the target budget   # data-driven start
for each step t:
    if sigma[t] > thr      -> model unsure  -> READ physical probe
    if t - last_read >= T_base -> safety cadence -> READ physical probe
    else -> use the virtual prediction
    # integral control: adjust thr toward the target usage rate
    rate = reads_so_far / steps_so_far
    thr  = thr * (1 + kp*(rate - budget))     # too many reads -> raise threshold
```
- **Integral control law:** if the probe is being used more than the budget, the
  threshold rises (becomes stricter), so fewer future reads are triggered — a
  self-correcting feedback loop.
- **Safety cadence (T_base):** always read occasionally even if confident.
- **Measured result:** asked for 5/10/20% usage → achieved 5.1/9.9/19.9% (within
  ~0.2%), and accuracy *rises* with the budget.

## The honest evaluation (`main`)
- Builds the dataset **with actuators** (sensors first, then dosing signals), on
  the **leakage-free temporal split**.
- Note on τ: the code comment explains persistence error ≈0.002 scaled, so
  τ=0.003 keeps a learning signal — this is the scientific insight above.
- Trains CARD, then compares against **persistence** and the **Improved GRU** on
  R²/MAE/RMSE/within-±0.05.
- Runs CASC at budgets {5,10,20%}.
- Checks the gate: correlation of `g` with actuator activity (should be positive).
- Saves `card_results.json`, `card_best.pt`, and `card_test_stream.npz`.

### Likely faculty questions on card_eval.py
- *"Is CASC learned or rule-based?"* → Rule-based feedback control (integral
  law), which is a strength: predictable, tunable, no extra training.
- *"How do you convert σ to pH units?"* → Multiply the scaled σ by the target
  MinMax range (`data_max - data_min`).
- *"Why compare to persistence?"* → It's the trivial baseline every model must
  beat; on this smooth signal it's near the ceiling, which we report honestly.
- *"Does CASC touch the test set unfairly?"* → No — it's an inference-time policy
  evaluated on the test stream; thresholds start from σ quantiles, adapted
  online.

---

# FILE 3 — `card_ablation.py` (which components actually help)

An **ablation study** removes one component at a time to measure its real
contribution — the gold standard for justifying a multi-part design.

Variants trained (same split, same seed):
| Variant | What's removed |
|---|---|
| A. Full CARD | nothing (baseline) |
| B. gate = 1 | the adaptive skip (becomes a plain delta head) |
| C. MSE loss | the Tolerance-Weighted Huber loss |
| D. no NLL term | the uncertainty-head training signal |
| E. no gate prior | the dosing-aware supervision of the gate |

Implementation detail: `train_variant` has flags (`force_gate_one`, `use_twh`,
`use_nll`, `use_gate_prior`) that switch each component on/off inside the loss,
so the architecture stays identical and only the studied piece changes — a
clean, controlled experiment.

**Measured findings (honest):**
- B (gate=1) lowers R² and tolerance → the adaptive skip helps a little.
- E (no gate prior) lowers tolerance and makes the gate uninterpretable.
- C (MSE) ties TWH on this near-ceiling data (no harm; TWH aligns the objective).
- D confirms the uncertainty head doesn't change point accuracy — its job is to
  **enable CASC**, not to improve prediction.

### Likely faculty questions on card_ablation.py
- *"What is an ablation study and why do it?"* → Systematically disabling parts
  to prove each contributes; prevents "it works but we don't know why."
- *"Which component matters most?"* → For point accuracy, the adaptive skip;
  for the system, the uncertainty head (it enables budgeted sampling via CASC).
- *"How do you keep it a fair test?"* → Same data split, same seed, same
  architecture; only the loss terms toggle.

---

# WRAP-UP: the honest story to tell

1. **Novelty = fusion:** gated residual forecasting + tolerance-weighted loss +
   self-supervised uncertainty + budget-targeting controller, for hydroponic
   virtual pH sensing.
2. **Honest result:** on this smooth signal CARD *matches* strong baselines on
   raw accuracy (persistence is near the ceiling) but **leads on the ±0.05
   tolerance metric** and adds capabilities baselines cannot: interpretable
   dosing gating (corr ≈ +0.98), calibrated uncertainty, and a controller that
   hits a chosen probe-usage budget within ~0.2%.
3. **Scientific insight:** the tolerance dead-zone τ must be smaller than the
   typical error, or the gradient vanishes — a practical rule for this loss.

*Files:* `src/card.py` (model + TWH loss + NLL + gate prior + trainer),
`src/card_eval.py` (CASC controller + evaluation), `src/card_ablation.py`
(component ablation). Results in `results/metrics/card_results.json` and
`card_ablation.json`; figures `results/figures/card_*.png`.
