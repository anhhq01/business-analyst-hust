from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def mase(y_true, y_pred, y_naive) -> float:
    model_mae = mean_absolute_error(y_true, y_pred)
    naive_mae = mean_absolute_error(y_true, y_naive)
    return float(model_mae / naive_mae) if naive_mae else np.nan


def regression_metrics(y_true, y_pred, y_naive=None) -> dict:
    out = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(mean_squared_error(y_true, y_pred) ** 0.5),
        'wape': wape(y_true, y_pred),
    }
    if y_naive is not None:
        out['mase_vs_seasonal_naive'] = mase(y_true, y_pred, y_naive)
    return out


def peak_metrics(y_true, y_pred, capacity_threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    actual_peak = y_true >= capacity_threshold
    predicted_peak = y_pred >= capacity_threshold
    tp = int(np.sum(actual_peak & predicted_peak))
    fp = int(np.sum(~actual_peak & predicted_peak))
    fn = int(np.sum(actual_peak & ~predicted_peak))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {
        'capacity_threshold': float(capacity_threshold),
        'actual_breaches': int(actual_peak.sum()),
        'predicted_breaches': int(predicted_peak.sum()),
        'breaches_correctly_anticipated': tp,
        'peak_precision': None if np.isnan(precision) else float(precision),
        'peak_recall': None if np.isnan(recall) else float(recall),
    }
