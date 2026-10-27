"""
Verify Component (1): does the dosing gate open more when actuators are active?

The paper-subset TEST window has no dosing (all actuators off), so we check the
gate on the FULL month (which contains real dosing events) and on the CARD
train+val windows, where actuator activity varies.
"""
import numpy as np
import torch

import dataset as ds
from card import CARD, CARDConfig, predict
from config import MODELS, SENSOR_FEATURES


def load_card(target_index):
    ck = torch.load(MODELS / "card_best.pt", map_location="cpu")
    m = CARD(CARDConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"]); m.eval()
    return m


def check(split_windows, n_sensor, model, label):
    out = predict(model, split_windows)
    gate = out["gate"]
    recent_act = split_windows[:, -1, n_sensor:].sum(axis=1)
    active = recent_act > 0
    quiet = ~active
    if active.sum() == 0:
        print(f"[{label}] no dosing activity present (all actuators off).")
        return
    corr = (np.corrcoef(gate, recent_act)[0, 1]
            if recent_act.std() > 0 else float("nan"))
    print(f"[{label}] windows={len(gate)}  active={int(active.sum())} "
          f"quiet={int(quiet.sum())}")
    print(f"         gate mean when DOSING active : {gate[active].mean():.4f}")
    print(f"         gate mean when system QUIET  : {gate[quiet].mean():.4f}")
    print(f"         corr(gate, actuator activity): {corr:.3f}")


def main():
    # CARD data layout: sensors then actuators
    wd_paper = ds.build(include_actuators=True, source="paper",
                        split="temporal", verbose=False)
    n_sensor = len(SENSOR_FEATURES)
    model = load_card(wd_paper.target_index)

    # paper subset: check train + val (these DO contain dosing events)
    check(wd_paper.X_train, n_sensor, model, "paper-subset TRAIN")
    check(wd_paper.X_val, n_sensor, model, "paper-subset VAL")
    check(wd_paper.X_test, n_sensor, model, "paper-subset TEST")

    # full month: richer dosing activity
    wd_full = ds.build(include_actuators=True, source="full",
                       split="temporal", verbose=False)
    check(wd_full.X_test, n_sensor, model, "full-month TEST")


if __name__ == "__main__":
    main()
