"""
Export the Improved GRU weights + MinMax scaler params to a JSON file so the
browser dashboard can run the EXACT same forward pass in JavaScript.

Output: results/web_model.json
Contains:
  - config (n_features, hidden, layers, delta_head, target_index)
  - LayerNorm gamma/beta
  - GRU weights for each layer (weight_ih, weight_hh, bias_ih, bias_hh)
  - head Linear1 (W,b), Linear2 (W,b)
  - feature scaler min_ / scale_  and target scaler min_/scale_
  - feature names, and a few example test windows (RAW pH-unit inputs) for a
    self-test / demo defaults.
Also verifies the exported weights reproduce PyTorch's output (parity check).
"""
from __future__ import annotations

import json

import numpy as np
import torch

import dataset as ds
from config import MODELS, RESULTS
from improved_gru import GRUConfig, ImprovedGRU


def t2l(x):
    return np.asarray(x.detach().cpu().numpy(), dtype=float).tolist()


def main():
    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    ck = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    cfg = GRUConfig(**ck["gru_cfg"])
    model = ImprovedGRU(cfg, target_index=wd.target_index)
    model.load_state_dict(ck["state"])
    model.eval()

    sd = model.state_dict()

    out = {
        "config": {
            "n_features": cfg.n_features,
            "hidden": cfg.hidden,
            "layers": cfg.layers,
            "delta_head": cfg.delta_head,
            "target_index": wd.target_index,
            "lookback": wd.X_test.shape[1],
        },
        "feature_names": wd.feature_names,
        "layernorm": {
            "gamma": t2l(sd["norm.weight"]) if "norm.weight" in sd else None,
            "beta": t2l(sd["norm.bias"]) if "norm.bias" in sd else None,
        },
        "gru_layers": [],
        "head": {
            "w1": t2l(sd["head.1.weight"]), "b1": t2l(sd["head.1.bias"]),
            "w2": t2l(sd["head.4.weight"]), "b2": t2l(sd["head.4.bias"]),
        },
        "feature_scaler": {
            "min": wd.feature_scaler.min_.tolist(),
            "scale": wd.feature_scaler.scale_.tolist(),
        },
        "target_scaler": {
            "min": float(wd.target_scaler.min_[0]),
            "scale": float(wd.target_scaler.scale_[0]),
            "data_min": float(wd.target_scaler.data_min_[0]),
            "data_max": float(wd.target_scaler.data_max_[0]),
        },
    }

    for L in range(cfg.layers):
        out["gru_layers"].append({
            "weight_ih": t2l(sd[f"gru.weight_ih_l{L}"]),
            "weight_hh": t2l(sd[f"gru.weight_hh_l{L}"]),
            "bias_ih": t2l(sd[f"gru.bias_ih_l{L}"]),
            "bias_hh": t2l(sd[f"gru.bias_hh_l{L}"]),
        })

    # A few example RAW windows (physical pH units) so the demo has real
    # starting values; store the median window and a couple of test windows.
    # Reconstruct raw feature values from the scaled X_test.
    fs_min = np.array(out["feature_scaler"]["min"])
    fs_scale = np.array(out["feature_scaler"]["scale"])
    raw_examples = []
    for i in [0, len(wd.X_test) // 2, len(wd.X_test) - 1]:
        scaled = wd.X_test[i]                    # (lookback, features)
        raw = (scaled - fs_min) / fs_scale       # invert MinMax
        raw_examples.append(raw.tolist())
    out["example_windows_raw"] = raw_examples

    # Parity self-check: run PyTorch vs a numpy re-implementation on 20 windows.
    with torch.no_grad():
        torch_pred = model(torch.from_numpy(wd.X_test[:20])).numpy()
    np_pred = np.array([_np_forward(out, wd.X_test[i]) for i in range(20)])
    max_diff = float(np.max(np.abs(torch_pred - np_pred)))
    out["parity_max_abs_diff"] = max_diff
    print(f"[parity] max|torch - numpy re-impl| = {max_diff:.2e}")

    path = RESULTS / "web_model.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[save] {path}  ({path.stat().st_size/1024:.0f} KB)")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_forward(m, x_scaled):
    """NumPy re-implementation of ImprovedGRU forward, for parity check.
    Mirrors exactly what the JS will do."""
    cfg = m["config"]
    h = np.array(x_scaled, dtype=float)  # (T, F) already scaled

    # LayerNorm over features (per time step)
    if m["layernorm"]["gamma"] is not None:
        g = np.array(m["layernorm"]["gamma"]); b = np.array(m["layernorm"]["beta"])
        mean = h.mean(axis=1, keepdims=True)
        var = h.var(axis=1, keepdims=True)
        h = (h - mean) / np.sqrt(var + 1e-5) * g + b

    seq = h
    hidden = cfg["hidden"]
    for L in range(cfg["layers"]):
        gl = m["gru_layers"][L]
        Wih = np.array(gl["weight_ih"]); Whh = np.array(gl["weight_hh"])
        bih = np.array(gl["bias_ih"]); bhh = np.array(gl["bias_hh"])
        ht = np.zeros(hidden)
        outs = []
        for t in range(seq.shape[0]):
            xt = seq[t]
            gi = Wih @ xt + bih      # (3H,)
            gh = Whh @ ht + bhh      # (3H,)
            i_r, i_z, i_n = gi[:hidden], gi[hidden:2*hidden], gi[2*hidden:]
            h_r, h_z, h_n = gh[:hidden], gh[hidden:2*hidden], gh[2*hidden:]
            r = _sigmoid(i_r + h_r)
            z = _sigmoid(i_z + h_z)
            n = np.tanh(i_n + r * h_n)
            ht = (1 - z) * n + z * ht
            outs.append(ht.copy())
        seq = np.array(outs)  # becomes input to next layer

    last = seq[-1]
    # head: Linear1 -> ReLU -> Linear2
    W1 = np.array(m["head"]["w1"]); b1 = np.array(m["head"]["b1"])
    W2 = np.array(m["head"]["w2"]); b2 = np.array(m["head"]["b2"])
    z1 = np.maximum(0.0, W1 @ last + b1)
    y = (W2 @ z1 + b2)[0]

    if cfg["delta_head"]:
        last_ph_scaled = x_scaled[-1, cfg["target_index"]]
        y = last_ph_scaled + y
    return y


if __name__ == "__main__":
    main()
