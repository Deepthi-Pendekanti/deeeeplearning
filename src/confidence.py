"""
Phase 7 — Confidence-aware virtual pH sensor.

Two complementary reliability signals:

1. MODEL UNCERTAINTY via MC-Dropout: keep dropout active at inference, run K
   stochastic forward passes, take the prediction std as an epistemic
   uncertainty estimate. High std => the model is unsure.

2. RUNNING ERROR (paper Algorithm 4): whenever the physical probe IS read,
   update an exponential moving average of |virtual - physical|. A rising EMA
   => the virtual sensor is drifting from reality.

Switching policy (hysteresis, like the paper):
  * Start virtual-dominant.
  * If (uncertainty > u_high) OR (EMA error > e_high): switch to
    physical-dominant and READ the physical probe (also refreshes the EMA).
  * If back below the low thresholds: return to virtual-dominant.
  * Always read physical at a slow baseline cadence (T_base) for safety.

We then MEASURE, on the test stream, what fraction of steps actually needed the
physical probe, and the resulting accuracy. No lifespan number is invented; we
report only the measured reduction in physical reads.
"""
from __future__ import annotations

import numpy as np
import torch


def mc_dropout_predict(model, X: np.ndarray, k: int = 30):
    """Return (mean_pred_scaled, std_pred_scaled) over k stochastic passes."""
    model.train()  # enable dropout
    xt = torch.from_numpy(X)
    preds = []
    with torch.no_grad():
        for _ in range(k):
            preds.append(model(xt).numpy())
    model.eval()
    arr = np.stack(preds, axis=0)  # (k, N)
    return arr.mean(0), arr.std(0)


def simulate_switching(true_ph: np.ndarray,
                       virt_ph: np.ndarray,
                       uncertainty: np.ndarray,
                       u_low: float, u_high: float,
                       e_low: float, e_high: float,
                       alpha: float = 0.3,
                       t_base: int = 60):
    """
    Walk the test stream one step at a time applying the hybrid policy.

    Returns a dict of measured outcomes:
      * physical_reads      : how many steps used the physical probe
      * physical_fraction   : reads / N
      * output_ph           : the pH the SYSTEM reports each step (hybrid)
      * mode                : 'virtual'/'physical' per step
    """
    n = len(true_ph)
    ema = 0.0
    mode = "virtual"
    last_phys = -10**9
    physical_reads = 0
    output = np.empty(n)
    modes = []

    for t in range(n):
        u = uncertainty[t]
        need_physical = False

        # baseline safety cadence
        if t - last_phys >= t_base:
            need_physical = True
        # trigger on high uncertainty or high running error
        if u > u_high or ema > e_high:
            need_physical = True
            mode = "physical"
        # relax back to virtual when comfortably low
        if u < u_low and ema < e_low:
            mode = "virtual"

        if need_physical:
            phys = true_ph[t]                 # read the real probe
            physical_reads += 1
            last_phys = t
            err = abs(phys - virt_ph[t])
            ema = alpha * err + (1 - alpha) * ema
            # in physical-dominant mode the probe overrides for control
            output[t] = phys if mode == "physical" else virt_ph[t]
        else:
            output[t] = virt_ph[t]            # trust the virtual sensor

        modes.append(mode)

    return {
        "physical_reads": int(physical_reads),
        "physical_fraction": physical_reads / n,
        "output_ph": output,
        "modes": modes,
    }
