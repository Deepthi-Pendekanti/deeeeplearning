"""Evaluation metrics (Section 3.3.9): RMSE (Eq. 23), MAE (Eq. 24), R2 (Eq. 25)."""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%), used in Table 6."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def within_error(y_true: np.ndarray, y_pred: np.ndarray, pct: float) -> float:
    """% of samples with relative error <= pct (e.g. 0.05 for 5%). Table 6."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    rel = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(rel <= pct) * 100)


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray,
                     tol: float = 0.05) -> float:
    """% of samples with absolute residual <= tol pH units (Sec 4.5)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= tol) * 100)


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute error (pH units) — robust to spike outliers."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def maxae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error (pH units) — worst-case single prediction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MedAE": medae(y_true, y_pred),
        "MaxAE": maxae(y_true, y_pred),
        # Absolute pH tolerance bands (the scientifically meaningful ones here).
        "within_0.01pH": within_tolerance(y_true, y_pred, 0.01),
        "within_0.03pH": within_tolerance(y_true, y_pred, 0.03),
        "within_0.05pH": within_tolerance(y_true, y_pred, 0.05),
        "within_0.10pH": within_tolerance(y_true, y_pred, 0.10),
        # Relative-error bands (kept for continuity with the paper's Table 6).
        "within_5pct": within_error(y_true, y_pred, 0.05),
        "within_10pct": within_error(y_true, y_pred, 0.10),
        "within_15pct": within_error(y_true, y_pred, 0.15),
    }
