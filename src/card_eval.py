"""
CARD evaluation + Component (4): Confidence-Adaptive Sampling Controller (CASC).

Trains CARD on the leakage-free temporal split, compares it to the Improved GRU
and persistence, and runs CASC: an online controller that adapts the uncertainty
threshold to hit a target physical-probe usage BUDGET while minimizing
tolerance (+/-0.05 pH) violations.

All numbers are measured. Test set is only touched for final reporting.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import dataset as ds
from card import CARD, CARDConfig, CARDTrainConfig, predict, train_card
from config import ACTUATOR_FEATURES, METRICS, MODELS, RESIDUAL_TOLERANCE, SENSOR_FEATURES
from metrics import all_metrics


# ---------------------------------------------------------------------------
# Build a windowed dataset with sensors FIRST then actuators (CARD needs both)
# ---------------------------------------------------------------------------
def build_card_data(source="paper", split="temporal"):
    # include_actuators=True appends actuators after the 6 sensors, so the
    # feature layout is [pH, ...sensors..., ...actuators...]; target_index=0.
    wd = ds.build(include_actuators=True, source=source, split=split,
                  verbose=False)
    n_sensor = len(SENSOR_FEATURES)
    n_actuator = len(wd.feature_names) - n_sensor
    return wd, n_sensor, n_actuator


# ---------------------------------------------------------------------------
# (4) CASC — Confidence-Adaptive Sampling Controller
# ---------------------------------------------------------------------------
def casc_controller(true_ph, virt_ph, sigma, budget=0.10,
                    tol=RESIDUAL_TOLERANCE, kp=0.5, t_base=120):
    """
    Walk the stream. Maintain an uncertainty threshold `thr`. Each step:
      * if sigma[t] > thr  (model unsure) OR safety cadence hit -> read physical.
      * adapt thr with a simple integral control law to track the target
        `budget` (fraction of steps that read physical).
    Returns measured usage, hybrid output pH, and per-step mode/threshold.
    """
    n = len(true_ph)
    # initialize threshold at the budget-quantile of sigma (data-driven start)
    thr = float(np.quantile(sigma, 1.0 - budget)) if n else 1.0
    reads = 0
    last_phys = -10**9
    out = np.empty(n)
    modes = []
    thr_trace = []
    used = 0

    for t in range(n):
        need = False
        if sigma[t] > thr:
            need = True
        if t - last_phys >= t_base:      # safety cadence
            need = True
        if need:
            out[t] = true_ph[t]          # physical read overrides
            reads += 1
            last_phys = t
            modes.append("physical")
        else:
            out[t] = virt_ph[t]
            modes.append("virtual")
        used += 1
        # integral control: nudge thr toward the target usage rate so far
        rate = reads / used
        thr = thr * (1.0 + kp * (rate - budget))   # too many reads -> raise thr
        thr = max(thr, 1e-6)
        thr_trace.append(thr)

    return {
        "physical_reads": int(reads),
        "physical_fraction": reads / n,
        "output_ph": out,
        "modes": modes,
        "thr_trace": thr_trace,
    }


# ---------------------------------------------------------------------------
# Reference: load the Improved GRU for a head-to-head
# ---------------------------------------------------------------------------
def improved_gru_test_pred():
    from improved_gru import GRUConfig, ImprovedGRU
    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    ck = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    m = ImprovedGRU(GRUConfig(**ck["gru_cfg"]), target_index=wd.target_index)
    m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        p = ds.inverse_target(wd, m(torch.from_numpy(wd.X_test)).numpy())
    return ds.inverse_target(wd, wd.y_test), p


def main():
    wd, n_sensor, n_actuator = build_card_data()
    print(f"[card] features={wd.feature_names}")
    print(f"[card] n_sensor={n_sensor} n_actuator={n_actuator}")

    cfg = CARDConfig(n_sensor=n_sensor, n_actuator=n_actuator, hidden=32,
                     layers=2, dropout=0.1, target_index=wd.target_index,
                     uncertainty_head=True)
    # NOTE: persistence error is ~0.002 (scaled); a large dead-zone (0.02) makes
    # ~99% of samples zero-loss and kills the gradient. We set tau just BELOW the
    # typical error so TWH ignores true sensor noise yet still learns structure.
    tc = CARDTrainConfig(epochs=120, tau=0.003, huber_delta=0.02,
                         nll_weight=0.2, gate_prior_weight=0.3,
                         patience=20, seed=42)
    model, hist = train_card(CARD(cfg), wd, tc, verbose=False)
    print(f"[card] trained: epochs={hist['epochs_run']} "
          f"best_val_twh={hist['best_val_twh']:.3e}")

    # ---- predictions on test (physical pH units) ----
    out = predict(model, wd.X_test)
    card_pred = ds.inverse_target(wd, out["pH_hat"])
    true = ds.inverse_target(wd, wd.y_test)
    # sigma scaled -> physical pH units
    scale = wd.target_scaler.data_max_[0] - wd.target_scaler.data_min_[0]
    sigma_ph = out.get("sigma", np.zeros_like(card_pred)) * scale
    gate = out["gate"]

    m_card = all_metrics(true, card_pred)

    # ---- baselines ----
    true_ig, ig_pred = improved_gru_test_pred()
    m_ig = all_metrics(true_ig, ig_pred)
    persist = ds.inverse_target(wd, wd.X_test[:, -1, wd.target_index])
    m_persist = all_metrics(true, persist)

    print("\n" + "=" * 70)
    print("CARD vs baselines — leakage-free temporal split (test, physical pH)")
    print("=" * 70)
    for name, m in [("Persistence", m_persist), ("Improved GRU", m_ig),
                    ("CARD (ours)", m_card)]:
        print(f"{name:<16} R2={m['R2']:.4f}  MAE={m['MAE']:.4f}  "
              f"RMSE={m['RMSE']:.4f}  within0.05={m['within_0.05pH']:.2f}%")

    # ---- (4) CASC at several usage budgets ----
    print("\nCASC controller — physical-probe usage budget vs achieved:")
    casc_rows = []
    for budget in [0.05, 0.10, 0.20]:
        sim = casc_controller(true, card_pred, sigma_ph, budget=budget)
        mh = all_metrics(true, sim["output_ph"])
        casc_rows.append({
            "budget": budget,
            "achieved_usage": sim["physical_fraction"],
            "hybrid_R2": mh["R2"], "hybrid_MAE": mh["MAE"],
            "hybrid_within_0.05pH": mh["within_0.05pH"],
        })
        print(f"  budget={budget*100:4.0f}%  achieved={sim['physical_fraction']*100:5.1f}%"
              f"  hybrid R2={mh['R2']:.4f}  within0.05={mh['within_0.05pH']:.2f}%")

    # gate sanity: does the gate open more during dosing activity?
    recent_act_sum = wd.X_test[:, -1, n_sensor:].sum(axis=1)
    if recent_act_sum.std() > 0:
        gate_corr = float(np.corrcoef(gate, recent_act_sum)[0, 1])
    else:
        gate_corr = float("nan")
    print(f"\n[gate] corr(gate, recent actuator activity) = {gate_corr:.3f} "
          f"(positive => gate opens during dosing, as designed)")

    result = {
        "card_metrics": m_card,
        "improved_gru_metrics": m_ig,
        "persistence_metrics": m_persist,
        "casc": casc_rows,
        "gate_actuator_corr": gate_corr,
        "train": {"epochs_run": hist["epochs_run"],
                  "best_val_twh": hist["best_val_twh"]},
        "config": {"tau": tc.tau, "huber_delta": tc.huber_delta,
                   "nll_weight": tc.nll_weight, "hidden": cfg.hidden,
                   "layers": cfg.layers},
    }
    with open(METRICS / "card_results.json", "w") as f:
        json.dump(result, f, indent=2)
    # save artifacts for plotting
    np.savez(MODELS / "card_test_stream.npz", true=true, card=card_pred,
             improved_gru=ig_pred, persistence=persist, sigma=sigma_ph,
             gate=gate, recent_act=recent_act_sum)
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__,
                "target_index": wd.target_index},
               MODELS / "card_best.pt")
    print("\n[save] results/metrics/card_results.json + card_best.pt + stream")


if __name__ == "__main__":
    main()
