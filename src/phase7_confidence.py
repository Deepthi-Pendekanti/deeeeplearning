"""
Phase 7 driver — confidence-aware hybrid virtual/physical pH sensing.

Loads the improved GRU (Phase 5), estimates MC-Dropout uncertainty, calibrates
switching thresholds from the VALIDATION residuals (test kept honest), runs the
hybrid switching simulation on the test stream, and reports:
  * measured physical-probe usage (and reduction vs always-on)
  * accuracy of the hybrid system output vs always-physical (ground truth)
"""
from __future__ import annotations

import json

import numpy as np
import torch

import dataset as ds
from config import METRICS, MODELS
from confidence import mc_dropout_predict, simulate_switching
from improved_gru import GRUConfig, ImprovedGRU
from metrics import all_metrics


def load_improved(nf, target_index):
    ckpt = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    gcfg = GRUConfig(**ckpt["gru_cfg"])
    model = ImprovedGRU(gcfg, target_index=target_index)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def main():
    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    model = load_improved(wd.n_features, wd.target_index)

    # --- Uncertainty + point predictions on val and test (physical units) ---
    vmean_s, vstd_s = mc_dropout_predict(model, wd.X_val, k=30)
    tmean_s, tstd_s = mc_dropout_predict(model, wd.X_test, k=30)

    val_pred = ds.inverse_target(wd, vmean_s)
    val_true = ds.inverse_target(wd, wd.y_val)
    test_pred = ds.inverse_target(wd, tmean_s)
    test_true = ds.inverse_target(wd, wd.y_test)

    # Uncertainty in physical pH units (scale std by target range).
    scale = (wd.target_scaler.data_max_[0] - wd.target_scaler.data_min_[0])
    val_unc = vstd_s * scale
    test_unc = tstd_s * scale

    # --- Calibrate thresholds from VALIDATION only ---
    # Uncertainty thresholds: 60th/90th percentile of val uncertainty.
    u_low = float(np.percentile(val_unc, 60))
    u_high = float(np.percentile(val_unc, 90))
    # Error EMA thresholds: tie to the +-0.05 pH tolerance band used throughout.
    e_low, e_high = 0.03, 0.05

    print(f"[phase7] thresholds  u_low={u_low:.4f} u_high={u_high:.4f} "
          f"e_low={e_low} e_high={e_high}")

    # --- Pure virtual (no switching) accuracy for reference ---
    m_virtual = all_metrics(test_true, test_pred)

    # --- Hybrid switching simulation on the test stream ---
    sim = simulate_switching(
        true_ph=test_true, virt_ph=test_pred, uncertainty=test_unc,
        u_low=u_low, u_high=u_high, e_low=e_low, e_high=e_high,
        alpha=0.3, t_base=60,
    )
    m_hybrid = all_metrics(test_true, sim["output_ph"])

    n = len(test_true)
    print("\n" + "=" * 64)
    print("Phase 7 — hybrid virtual/physical pH sensing (test stream)")
    print("=" * 64)
    print(f"test steps                : {n}")
    print(f"physical reads (hybrid)   : {sim['physical_reads']} "
          f"({100*sim['physical_fraction']:.1f}% of steps)")
    print(f"physical reads (always-on): {n} (100%)")
    print(f"REDUCTION in physical use : "
          f"{100*(1-sim['physical_fraction']):.1f}%")
    print(f"\npure-virtual  : R2={m_virtual['R2']:.4f} MAE={m_virtual['MAE']:.4f} "
          f"within0.05={m_virtual['within_0.05pH']:.2f}%")
    print(f"hybrid system : R2={m_hybrid['R2']:.4f} MAE={m_hybrid['MAE']:.4f} "
          f"within0.05={m_hybrid['within_0.05pH']:.2f}%")
    print("(always-physical is exact by definition — the reference ground truth)")

    out = {
        "thresholds": {"u_low": u_low, "u_high": u_high,
                       "e_low": e_low, "e_high": e_high},
        "test_steps": n,
        "physical_reads": sim["physical_reads"],
        "physical_fraction": sim["physical_fraction"],
        "reduction_pct": 100 * (1 - sim["physical_fraction"]),
        "pure_virtual_metrics": m_virtual,
        "hybrid_metrics": m_hybrid,
    }
    np.savez(MODELS / "phase7_hybrid_stream.npz",
             true=test_true, virtual=test_pred, hybrid=sim["output_ph"],
             uncertainty=test_unc,
             physical_mask=np.array([m == "physical" for m in sim["modes"]]))
    with open(METRICS / "phase7_confidence.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] results/metrics/phase7_confidence.json")


if __name__ == "__main__":
    main()
