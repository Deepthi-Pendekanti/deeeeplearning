"""
Phase 5 — Systematic GRU improvement (validation-driven, test untouched).

Protocol:
  * Split = temporal (leakage-free) -- this is the yardstick that matters.
  * Model/train configs are scored on the VALIDATION set only.
  * The single best-by-val config is then evaluated ONCE on the test set and
    compared against the reproduced GRU baseline and persistence.

Search space (small, sensible, reproducible -- not random flailing):
  * delta_head (predict pH change) : {True, False}
  * loss                           : {huber, mse}
  * hidden                         : {32, 64}
  * layers                         : {1, 2}
  * dropout                        : {0.1, 0.3}
  * input_layernorm                : {True}
  * weight_decay                   : {1e-4}
  * lr                             : {1e-3}
Kept deliberately compact so it runs on CPU in reasonable time.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

import dataset as ds
from config import METRICS, MODELS
from improved_gru import GRUConfig, ImprovedGRU, TrainConfig, train_improved
from metrics import all_metrics
from models import count_parameters


def eval_on(model, wd, which):
    X = getattr(wd, f"X_{which}")
    y = getattr(wd, f"y_{which}")
    model.eval()
    with torch.no_grad():
        pred = ds.inverse_target(wd, model(torch.from_numpy(X)).numpy())
    true = ds.inverse_target(wd, y)
    return true, pred


def persistence(wd, which="test"):
    X = getattr(wd, f"X_{which}")
    y = getattr(wd, f"y_{which}")
    pred = ds.inverse_target(wd, X[:, -1, wd.target_index])
    true = ds.inverse_target(wd, y)
    return all_metrics(true, pred)


def main():
    source, split = "paper", "temporal"
    wd = ds.build(include_actuators=False, source=source, split=split,
                  verbose=True)
    nf = wd.n_features

    grid = list(itertools.product(
        [True, False],       # delta_head
        ["huber", "mse"],    # loss
        [32, 64],            # hidden
        [1, 2],              # layers
        [0.1, 0.3],          # dropout
    ))
    print(f"[phase5] searching {len(grid)} configs on VALIDATION (temporal split)")

    trials = []
    best = None  # (val_metric, cfg_dict, model)
    for i, (delta, loss, hidden, layers, dropout) in enumerate(grid, 1):
        gcfg = GRUConfig(n_features=nf, hidden=hidden, layers=layers,
                         dropout=dropout, input_layernorm=True,
                         delta_head=delta)
        tcfg = TrainConfig(epochs=100, batch=32, lr=1e-3, weight_decay=1e-4,
                           optimizer="adamw", loss=loss, huber_delta=0.01,
                           grad_clip=1.0, patience=12, scheduler=True, seed=42)
        model = ImprovedGRU(gcfg, target_index=wd.target_index)
        model, hist = train_improved(model, wd, tcfg, verbose=False)

        # Select by VALIDATION metrics (physical pH), never test.
        vt, vp = eval_on(model, wd, "val")
        vm = all_metrics(vt, vp)
        rec = {"delta_head": delta, "loss": loss, "hidden": hidden,
               "layers": layers, "dropout": dropout,
               "val_R2": vm["R2"], "val_MAE": vm["MAE"], "val_RMSE": vm["RMSE"],
               "epochs_run": hist["epochs_run"], "params": count_parameters(model)}
        trials.append(rec)
        # Select by lowest validation MAE (robust, scale-meaningful).
        score = vm["MAE"]
        if best is None or score < best[0]:
            best = (score, rec, model, gcfg, tcfg)
        print(f"  [{i:2d}/{len(grid)}] delta={delta} loss={loss} h={hidden} "
              f"L={layers} do={dropout} -> val R2={vm['R2']:.3f} "
              f"MAE={vm['MAE']:.4f}")

    _, bestrec, bestmodel, bestg, bestt = best
    print(f"\n[phase5] BEST by val MAE: {bestrec}")

    # ---- Evaluate the winner ONCE on the test set ----
    tt, tp = eval_on(bestmodel, wd, "test")
    test_m = all_metrics(tt, tp)
    test_m["params"] = count_parameters(bestmodel)

    persist_m = persistence(wd, "test")

    print("\n" + "=" * 60)
    print("Phase 5 — improved GRU on TEST (temporal, leakage-free)")
    print("=" * 60)
    for label, m in [("Persistence", persist_m),
                     ("Improved GRU", test_m)]:
        print(f"{label:<15} R2={m['R2']:.4f} MAE={m['MAE']:.4f} "
              f"RMSE={m['RMSE']:.4f} within0.05={m['within_0.05pH']:.2f}%")

    # Save winner + predictions + full trial log.
    torch.save({"state": bestmodel.state_dict(),
                "gru_cfg": bestg.__dict__, "train_cfg": bestt.__dict__,
                "target_index": wd.target_index, "n_features": nf},
               MODELS / "improved_gru_best.pt")
    np.savez(MODELS / "improved_gru_test_predictions.npz", y_true=tt, y_pred=tp)
    with open(METRICS / "phase5_search.json", "w") as f:
        json.dump({"split": split, "source": source, "trials": trials,
                   "best_config": bestrec, "test_metrics": test_m,
                   "persistence_test": persist_m}, f, indent=2)
    print(f"\n[save] results/metrics/phase5_search.json + improved_gru_best.pt")


if __name__ == "__main__":
    main()
