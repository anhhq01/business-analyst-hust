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


def pinball_loss(y_true, y_pred_q, alpha: float) -> float:
    """Compute quantile (pinball) loss at quantile level alpha in (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    diff = y_true - y_pred_q
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def interval_metrics(
    y_true,
    y_lower,
    y_upper,
    nominal_coverage: float = 0.80,
) -> dict:
    """Evaluate prediction interval empirical coverage, width, and Winkler score."""
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.maximum(np.asarray(y_upper, dtype=float), y_lower)

    inside = (y_true >= y_lower) & (y_true <= y_upper)
    below = y_true < y_lower
    above = y_true > y_upper
    width = y_upper - y_lower

    alpha = 1.0 - nominal_coverage
    penalty_low = (2.0 / alpha) * (y_lower - y_true) * below
    penalty_high = (2.0 / alpha) * (y_true - y_upper) * above
    winkler = width + penalty_low + penalty_high

    emp_cov = float(np.mean(inside))
    return {
        'nominal_coverage': float(nominal_coverage),
        'empirical_coverage': emp_cov,
        'coverage_error': float(emp_cov - nominal_coverage),
        'below_lower_rate': float(np.mean(below)),
        'above_upper_rate': float(np.mean(above)),
        'mean_interval_width': float(np.mean(width)),
        'winkler_score': float(np.mean(winkler)),
    }


def probabilistic_metrics(y_true, pred_p10, pred_p50, pred_p90) -> dict:
    """Compute pinball loss across P10/P50/P90 and 80% interval coverage metrics."""
    pl_10 = pinball_loss(y_true, pred_p10, 0.10)
    pl_50 = pinball_loss(y_true, pred_p50, 0.50)
    pl_90 = pinball_loss(y_true, pred_p90, 0.90)
    iv = interval_metrics(y_true, pred_p10, pred_p90, nominal_coverage=0.80)
    return {
        'pinball_p10': pl_10,
        'pinball_p50': pl_50,
        'pinball_p90': pl_90,
        'mean_pinball_loss': float((pl_10 + pl_50 + pl_90) / 3.0),
        **iv,
    }


def regression_metrics(y_true, y_pred, y_naive=None) -> dict:
    out = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(mean_squared_error(y_true, y_pred) ** 0.5),
        'wape': wape(y_true, y_pred),
    }
    if y_naive is not None:
        out['mase_vs_seasonal_naive'] = mase(y_true, y_pred, y_naive)
    return out


def daily_peak_metrics(timestamps, y_true, y_pred) -> dict:
    """Evaluate daily peak magnitude error and daily peak timing error (in hours)."""
    frame = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'y_true': np.asarray(y_true, dtype=float),
        'y_pred': np.asarray(y_pred, dtype=float),
    })
    frame['date'] = frame['timestamp'].dt.date
    frame['hour'] = frame['timestamp'].dt.hour

    daily_records = []
    for _, grp in frame.groupby('date'):
        if len(grp) < 12:
            continue
        idx_true = grp['y_true'].idxmax()
        idx_pred = grp['y_pred'].idxmax()
        true_peak_val = float(grp.loc[idx_true, 'y_true'])
        pred_at_true_peak = float(grp.loc[idx_true, 'y_pred'])
        pred_peak_val = float(grp.loc[idx_pred, 'y_pred'])
        true_peak_hour = int(grp.loc[idx_true, 'hour'])
        pred_peak_hour = int(grp.loc[idx_pred, 'hour'])

        raw_hour_diff = abs(pred_peak_hour - true_peak_hour)
        timing_err = min(raw_hour_diff, 24 - raw_hour_diff)

        daily_records.append({
            'true_peak_val': true_peak_val,
            'pred_peak_val': pred_peak_val,
            'pred_at_true_peak': pred_at_true_peak,
            'timing_err_hours': timing_err,
        })

    if not daily_records:
        return {
            'daily_peak_mae': np.nan,
            'daily_peak_wape': np.nan,
            'coincident_peak_mae': np.nan,
            'coincident_peak_underprediction_rate': np.nan,
            'mean_abs_peak_timing_error_hours': np.nan,
            'exact_peak_hour_match_rate': np.nan,
            'within_1h_peak_timing_rate': np.nan,
        }

    d = pd.DataFrame(daily_records)
    return {
        'daily_peak_mae': float(mean_absolute_error(d['true_peak_val'], d['pred_peak_val'])),
        'daily_peak_wape': wape(d['true_peak_val'], d['pred_peak_val']),
        'coincident_peak_mae': float(mean_absolute_error(d['true_peak_val'], d['pred_at_true_peak'])),
        'coincident_peak_underprediction_rate': float(np.mean(d['pred_at_true_peak'] < d['true_peak_val'])),
        'mean_abs_peak_timing_error_hours': float(d['timing_err_hours'].mean()),
        'exact_peak_hour_match_rate': float((d['timing_err_hours'] == 0).mean()),
        'within_1h_peak_timing_rate': float((d['timing_err_hours'] <= 1).mean()),
    }


def peak_metrics(y_true, y_pred, capacity_threshold: float, timestamps=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual_peak = y_true >= capacity_threshold
    predicted_peak = y_pred >= capacity_threshold
    tp = int(np.sum(actual_peak & predicted_peak))
    fp = int(np.sum(~actual_peak & predicted_peak))
    fn = int(np.sum(actual_peak & ~predicted_peak))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else np.nan

    # Exceedance severity above threshold
    actual_exceedance = np.maximum(0.0, y_true - capacity_threshold)
    pred_exceedance = np.maximum(0.0, y_pred - capacity_threshold)
    unmet_peak_volume = float(np.maximum(0.0, actual_exceedance - pred_exceedance).sum())

    out = {
        'capacity_threshold': float(capacity_threshold),
        'actual_breaches': int(actual_peak.sum()),
        'predicted_breaches': int(predicted_peak.sum()),
        'breaches_correctly_anticipated': tp,
        'false_alarms': fp,
        'missed_breaches': fn,
        'peak_precision': None if np.isnan(precision) else float(precision),
        'peak_recall': None if np.isnan(recall) else float(recall),
        'peak_f1': None if np.isnan(f1) else float(f1),
        'unmet_exceedance_volume': unmet_peak_volume,
    }
    if timestamps is not None:
        out.update(daily_peak_metrics(timestamps, y_true, y_pred))
    return out


def evaluate_threshold_sensitivity(
    test_df: pd.DataFrame,
    train_load: pd.Series,
    percentiles: list[float] | tuple[float, ...] = (0.80, 0.90, 0.95, 0.975),
    reserve_margins: list[float] | tuple[float, ...] = (0.0, 0.05, 0.10, 0.15),
    unmet_penalty_per_unit: float = 10.0,
    reserve_cost_per_unit: float = 1.0,
) -> pd.DataFrame:
    """Simulate capacity threshold & reserve-margin rules across point and quantile policies.

    Scenario assumptions are explicit and non-causal:
    - If dispatched capacity = forecast * (1 + reserve_margin) exceeds `threshold`,
      operator prepares extra reserve `max(0, dispatched - threshold)`.
    - If actual load exceeds `max(threshold, dispatched)`, the shortfall incurs
      an illustrative `unmet_penalty_per_unit` (default 10:1 vs standby reserve cost).
    """
    y_true = test_df['system_load'].to_numpy(dtype=float)
    policies = {
        'LightGBM Point (L1)': test_df['prediction_lgbm'].to_numpy(dtype=float),
        'Seasonal Naive (168h)': test_df['prediction_seasonal_naive'].to_numpy(dtype=float),
    }
    if 'prediction_lgbm_p90' in test_df.columns:
        policies['LightGBM Quantile (P90)'] = test_df['prediction_lgbm_p90'].to_numpy(dtype=float)
    if 'prediction_ridge' in test_df.columns:
        policies['Ridge Regression'] = test_df['prediction_ridge'].to_numpy(dtype=float)

    rows = []
    for pct in percentiles:
        thresh = float(train_load.quantile(pct))
        actual_breach_mask = y_true >= thresh
        n_actual = int(actual_breach_mask.sum())

        for policy_name, pred in policies.items():
            pm = peak_metrics(y_true, pred, thresh)
            for margin in reserve_margins:
                dispatched = pred * (1.0 + margin)
                alert_mask = dispatched >= thresh
                tp = int(np.sum(actual_breach_mask & alert_mask))
                fp = int(np.sum(~actual_breach_mask & alert_mask))
                fn = int(np.sum(actual_breach_mask & ~alert_mask))
                recall = tp / n_actual if n_actual > 0 else np.nan
                precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

                extra_reserve = np.maximum(0.0, dispatched - thresh)
                effective_cap = thresh + extra_reserve
                unmet_load = np.maximum(0.0, y_true - effective_cap)

                reserve_cost = float(extra_reserve.sum() * reserve_cost_per_unit)
                shortfall_cost = float(unmet_load.sum() * unmet_penalty_per_unit)
                total_scenario_cost = reserve_cost + shortfall_cost

                rows.append({
                    'threshold_percentile': f'P{pct*100:g}',
                    'capacity_threshold': round(thresh, 2),
                    'policy': policy_name,
                    'reserve_margin_pct': round(margin * 100, 1),
                    'actual_breaches': n_actual,
                    'triggered_alerts': int(alert_mask.sum()),
                    'correct_alerts_tp': tp,
                    'false_alerts_fp': fp,
                    'missed_breaches_fn': fn,
                    'peak_recall': None if np.isnan(recall) else round(float(recall), 4),
                    'peak_precision': None if np.isnan(precision) else round(float(precision), 4),
                    'base_model_recall_0pct': pm['peak_recall'],
                    'max_extra_reserve_units': round(float(extra_reserve.max()), 1),
                    'total_extra_reserve_units': round(float(extra_reserve.sum()), 1),
                    'unmet_peak_units': round(float(unmet_load.sum()), 1),
                    'scenario_reserve_cost': round(reserve_cost, 1),
                    'scenario_shortfall_cost': round(shortfall_cost, 1),
                    'scenario_total_cost_10to1': round(total_scenario_cost, 1),
                })

    return pd.DataFrame(rows)
