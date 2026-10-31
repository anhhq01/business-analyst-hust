from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

from config import (
    RAW_DATA, PROCESSED_DATA, PREDICTIONS, METRICS, MODEL_PATH,
    MODEL_P10_PATH, MODEL_P50_PATH, MODEL_P90_PATH,
    BACKTEST_METRICS, BACKTEST_PREDICTIONS, SENSITIVITY_CSV,
    HORIZON_METRICS, ZONE_METRICS, SHAP_SUMMARY_CSV, SHAP_LOCAL_JSON,
    TARGET, TEST_DAYS, VALID_DAYS, RANDOM_STATE, WEATHER_COLS, ZONE_COLS,
    MULTI_HORIZONS, CAPACITY_PERCENTILES, RESERVE_MARGINS,
)
from preprocessing import build_horizon_table
from evaluation import (
    regression_metrics, peak_metrics, probabilistic_metrics,
    evaluate_threshold_sensitivity,
)

EXCLUDE = {
    'timestamp', TARGET, 'seasonal_naive_168h', 'seasonal_naive_24h',
    'observations_per_hour', 'is_complete_hour',
    *WEATHER_COLS, *ZONE_COLS,
}


def temporal_split(df: pd.DataFrame, test_days: int = TEST_DAYS, valid_days: int = VALID_DAYS):
    df = df.sort_values('timestamp').copy()
    test_start = df['timestamp'].max() - pd.to_timedelta((test_days * 24) - 1, unit='h')
    valid_start = test_start - pd.to_timedelta(valid_days * 24, unit='h')
    train = df[df['timestamp'] < valid_start].copy()
    valid = df[(df['timestamp'] >= valid_start) & (df['timestamp'] < test_start)].copy()
    test = df[df['timestamp'] >= test_start].copy()
    return train, valid, test


def feature_columns(df: pd.DataFrame, extra_exclude: set[str] | None = None) -> list[str]:
    excl = EXCLUDE if extra_exclude is None else (EXCLUDE | set(extra_exclude))
    return [c for c in df.columns if c not in excl]


def _make_lgbm(objective: str = 'regression_l1', alpha: float | None = None) -> lgb.LGBMRegressor:
    params = dict(
        objective=objective,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    if alpha is not None:
        params['alpha'] = alpha
    return lgb.LGBMRegressor(**params)


def fit_quantile_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET,
) -> tuple[dict[float, lgb.LGBMRegressor], float]:
    """Train LightGBM quantile models (P10, P50, P90) and compute split-conformal calibration on valid."""
    q_models: dict[float, lgb.LGBMRegressor] = {}
    for q in (0.10, 0.50, 0.90):
        qm = _make_lgbm(objective='quantile', alpha=q)
        qm.fit(
            train[features], train[target_col],
            eval_set=[(valid[features], valid[target_col])],
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        q_models[q] = qm

    # Conformalized Quantile Regression (CQR) adjustment calibrated strictly on validation set (80% target)
    val_p10 = q_models[0.10].predict(valid[features])
    val_p50 = q_models[0.50].predict(valid[features])
    val_p90 = q_models[0.90].predict(valid[features])
    val_low = np.minimum(val_p10, val_p50)
    val_high = np.maximum(val_p90, val_p50)
    y_val = valid[target_col].to_numpy(dtype=float)
    nonconformity = np.maximum(val_low - y_val, y_val - val_high)
    n_val = len(y_val)
    q_level = min(1.0, np.ceil((n_val + 1) * 0.80) / n_val)
    cqr_pad = float(np.quantile(nonconformity, q_level))
    return q_models, cqr_pad


def predict_quantiles(
    q_models: dict[float, lgb.LGBMRegressor],
    df: pd.DataFrame,
    features: list[str],
    cqr_pad: float = 0.0,
) -> dict[str, np.ndarray]:
    raw_10 = q_models[0.10].predict(df[features])
    raw_50 = q_models[0.50].predict(df[features])
    raw_90 = q_models[0.90].predict(df[features])

    # Enforce monotonicity p10 <= p50 <= p90
    p50 = raw_50
    p10_raw = np.minimum(raw_10, p50)
    p90_raw = np.maximum(raw_90, p50)

    # Validation-calibrated CQR interval
    p10_cal = np.minimum(p10_raw - cqr_pad, p50)
    p90_cal = np.maximum(p90_raw + cqr_pad, p50)
    return {
        'p10_raw': p10_raw,
        'p50': p50,
        'p90_raw': p90_raw,
        'p10_cal': p10_cal,
        'p90_cal': p90_cal,
    }


def fit_ridge_benchmark(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET,
) -> Pipeline:
    """Fit a leakage-safe standardized Ridge regression benchmark (BA6 framework)."""
    train_val = pd.concat([train, valid], axis=0).sort_values('timestamp')
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-2, 4, 25), cv=TimeSeriesSplit(n_splits=5))),
    ])
    pipe.fit(train_val[features], train_val[target_col])
    return pipe


def compute_explainability(
    model: lgb.LGBMRegressor,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET,
) -> dict:
    """Compute exact TreeSHAP values and permutation importance on the held-out test set."""
    X_test = test_df[features]
    y_test = test_df[target_col].to_numpy(dtype=float)

    # Native LightGBM exact TreeSHAP contributions: last column is expected value (base_value)
    contrib = model.booster_.predict(X_test, pred_contrib=True)
    shap_values = contrib[:, :-1]
    base_value = float(contrib[0, -1])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Permutation importance on test set (BA6 metric: increase in MAE when shuffled)
    perm = permutation_importance(
        model, X_test, y_test,
        scoring='neg_mean_absolute_error',
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    gain_imp = model.booster_.feature_importance(importance_type='gain')
    summary_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap,
        'permutation_mae_increase': perm.importances_mean,
        'permutation_mae_std': perm.importances_std,
        'lgbm_gain': gain_imp,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    SHAP_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SHAP_SUMMARY_CSV, index=False)

    # Local explanations for top 3 peak load hours in the test period
    top_peak_indices = np.argsort(y_test)[::-1][:3]
    local_explanations = []
    for idx in top_peak_indices:
        row_shap = shap_values[idx]
        top_feat_idx = np.argsort(np.abs(row_shap))[::-1][:10]
        local_explanations.append({
            'timestamp': str(test_df.iloc[idx]['timestamp']),
            'actual_load': float(y_test[idx]),
            'predicted_load': float(base_value + row_shap.sum()),
            'base_value': base_value,
            'top_contributions': [
                {
                    'feature': features[j],
                    'feature_value': float(X_test.iloc[idx, j]),
                    'shap_contribution': float(row_shap[j]),
                }
                for j in top_feat_idx
            ],
        })

    SHAP_LOCAL_JSON.write_text(json.dumps(local_explanations, indent=2), encoding='utf-8')
    return {
        'base_value': base_value,
        'top_10_shap_features': summary_df.head(10).to_dict(orient='records'),
    }


def run_rolling_backtest(df: pd.DataFrame, n_folds: int = 5, fold_days: int = 30) -> dict:
    """Run expanding-window rolling-origin backtesting across `n_folds` monthly windows."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    features = feature_columns(df)
    max_ts = df['timestamp'].max()

    fold_metrics = []
    fold_preds = []

    for k in range(n_folds, 0, -1):
        fold_id = n_folds - k + 1
        test_end = max_ts - pd.Timedelta(days=(k - 1) * fold_days)
        test_start = test_end - pd.Timedelta(days=fold_days) + pd.Timedelta(hours=1)
        valid_start = test_start - pd.Timedelta(days=14)

        train_f = df[df['timestamp'] < valid_start]
        valid_f = df[(df['timestamp'] >= valid_start) & (df['timestamp'] < test_start)]
        test_f = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()

        model_f = _make_lgbm('regression_l1')
        model_f.fit(
            train_f[features], train_f[TARGET],
            eval_set=[(valid_f[features], valid_f[TARGET])],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        q_models_f, cqr_pad_f = fit_quantile_models(train_f, valid_f, features, TARGET)
        q_preds_f = predict_quantiles(q_models_f, test_f, features, cqr_pad_f)
        ridge_f = fit_ridge_benchmark(train_f, valid_f, features, TARGET)

        pred_lgbm = model_f.predict(test_f[features])
        pred_ridge = ridge_f.predict(test_f[features])
        pred_naive = test_f['seasonal_naive_168h'].to_numpy(dtype=float)

        train_full_load = pd.concat([train_f[TARGET], valid_f[TARGET]])
        thresh_80 = float(train_full_load.quantile(0.80))

        reg_lgbm = regression_metrics(test_f[TARGET], pred_lgbm, pred_naive)
        reg_ridge = regression_metrics(test_f[TARGET], pred_ridge, pred_naive)
        reg_naive = regression_metrics(test_f[TARGET], pred_naive)
        prob_cal = probabilistic_metrics(
            test_f[TARGET], q_preds_f['p10_cal'], q_preds_f['p50'], q_preds_f['p90_cal']
        )
        pk_lgbm = peak_metrics(test_f[TARGET], pred_lgbm, thresh_80, timestamps=test_f['timestamp'])

        fold_metrics.append({
            'fold': fold_id,
            'test_start': str(test_f['timestamp'].min()),
            'test_end': str(test_f['timestamp'].max()),
            'train_rows': int(len(train_f) + len(valid_f)),
            'test_rows': int(len(test_f)),
            'capacity_threshold_p80': thresh_80,
            'lightgbm_mae': reg_lgbm['mae'],
            'lightgbm_rmse': reg_lgbm['rmse'],
            'lightgbm_wape': reg_lgbm['wape'],
            'lightgbm_mase': reg_lgbm['mase_vs_seasonal_naive'],
            'ridge_wape': reg_ridge['wape'],
            'ridge_mase': reg_ridge['mase_vs_seasonal_naive'],
            'seasonal_naive_wape': reg_naive['wape'],
            'mean_pinball_loss': prob_cal['mean_pinball_loss'],
            'empirical_coverage_80': prob_cal['empirical_coverage'],
            'peak_recall_p80': pk_lgbm['peak_recall'],
            'peak_precision_p80': pk_lgbm['peak_precision'],
            'daily_peak_wape': pk_lgbm['daily_peak_wape'],
            'mean_abs_peak_timing_error_hours': pk_lgbm['mean_abs_peak_timing_error_hours'],
        })

        fold_df = pd.DataFrame({
            'fold': fold_id,
            'timestamp': test_f['timestamp'],
            'system_load': test_f[TARGET],
            'prediction_lgbm': pred_lgbm,
            'prediction_ridge': pred_ridge,
            'prediction_seasonal_naive': pred_naive,
            'prediction_lgbm_p10': q_preds_f['p10_cal'],
            'prediction_lgbm_p50': q_preds_f['p50'],
            'prediction_lgbm_p90': q_preds_f['p90_cal'],
            'capacity_threshold': thresh_80,
        })
        fold_preds.append(fold_df)

    all_preds_df = pd.concat(fold_preds, ignore_index=True)
    BACKTEST_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    all_preds_df.to_csv(BACKTEST_PREDICTIONS, index=False)

    fm_df = pd.DataFrame(fold_metrics)
    summary = {
        'n_folds': n_folds,
        'fold_days': fold_days,
        'folds': fold_metrics,
        'aggregate': {
            'lightgbm_wape_mean': float(fm_df['lightgbm_wape'].mean()),
            'lightgbm_wape_std': float(fm_df['lightgbm_wape'].std()),
            'lightgbm_mase_mean': float(fm_df['lightgbm_mase'].mean()),
            'ridge_wape_mean': float(fm_df['ridge_wape'].mean()),
            'seasonal_naive_wape_mean': float(fm_df['seasonal_naive_wape'].mean()),
            'empirical_coverage_80_mean': float(fm_df['empirical_coverage_80'].mean()),
            'mean_pinball_loss_mean': float(fm_df['mean_pinball_loss'].mean()),
            'daily_peak_wape_mean': float(fm_df['daily_peak_wape'].mean()),
            'mean_abs_peak_timing_error_hours': float(fm_df['mean_abs_peak_timing_error_hours'].mean()),
        },
    }
    BACKTEST_METRICS.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_multi_horizon_study(raw_path: Path | str = RAW_DATA) -> dict:
    """Compare 1-hour, 6-hour, and 24-hour (day-ahead) forecasting performance."""
    results = {}
    for h in MULTI_HORIZONS:
        if h == 1 and PROCESSED_DATA.exists():
            df_h = pd.read_csv(PROCESSED_DATA, parse_dates=['timestamp'])
        else:
            df_h = build_horizon_table(path=raw_path, target_col=TARGET, horizon=h)
        train_h, valid_h, test_h = temporal_split(df_h)
        feats_h = feature_columns(df_h)

        model_h = _make_lgbm('regression_l1')
        model_h.fit(
            train_h[feats_h], train_h[TARGET],
            eval_set=[(valid_h[feats_h], valid_h[TARGET])],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        ridge_h = fit_ridge_benchmark(train_h, valid_h, feats_h, TARGET)
        q_models_h, cqr_pad_h = fit_quantile_models(train_h, valid_h, feats_h, TARGET)
        q_preds_h = predict_quantiles(q_models_h, test_h, feats_h, cqr_pad_h)

        pred_lgbm = model_h.predict(test_h[feats_h])
        pred_ridge = ridge_h.predict(test_h[feats_h])
        pred_naive = test_h['seasonal_naive_168h'].to_numpy(dtype=float)

        thresh = float(train_h[TARGET].quantile(0.80))
        results[f'{h}h_ahead'] = {
            'horizon_hours': h,
            'n_features': len(feats_h),
            'min_feature_lag_hours': h,
            'lightgbm': regression_metrics(test_h[TARGET], pred_lgbm, pred_naive),
            'ridge': regression_metrics(test_h[TARGET], pred_ridge, pred_naive),
            'seasonal_naive_168h': regression_metrics(test_h[TARGET], pred_naive),
            'probabilistic': probabilistic_metrics(
                test_h[TARGET], q_preds_h['p10_cal'], q_preds_h['p50'], q_preds_h['p90_cal']
            ),
            'peak_metrics': peak_metrics(test_h[TARGET], pred_lgbm, thresh, timestamps=test_h['timestamp']),
        }

    HORIZON_METRICS.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def run_zone_comparison_study(raw_path: Path | str = RAW_DATA) -> dict:
    """Compare zone-specific models (Zone 1, Zone 2, Zone 3 summed) vs pooled system_load model."""
    zone_preds_test = {}
    zone_naive_test = {}
    zone_reports = {}
    timestamps = None
    actual_system_test = None
    train_system_series = None

    for z_col in ZONE_COLS:
        df_z = build_horizon_table(path=raw_path, target_col=z_col, horizon=1)
        train_z, valid_z, test_z = temporal_split(df_z)
        feats_z = feature_columns(df_z)

        model_z = _make_lgbm('regression_l1')
        model_z.fit(
            train_z[feats_z], train_z[z_col],
            eval_set=[(valid_z[feats_z], valid_z[z_col])],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        pred_z = model_z.predict(test_z[feats_z])
        naive_z = test_z['seasonal_naive_168h'].to_numpy(dtype=float)
        thresh_z = float(train_z[z_col].quantile(0.80))

        zone_preds_test[z_col] = pred_z
        zone_naive_test[z_col] = naive_z
        zone_reports[z_col] = {
            'mean_load_test': float(test_z[z_col].mean()),
            'share_of_system_load_pct': float(test_z[z_col].sum() / test_z[TARGET].sum() * 100.0),
            'lightgbm': regression_metrics(test_z[z_col], pred_z, naive_z),
            'seasonal_naive': regression_metrics(test_z[z_col], naive_z),
            'peak_metrics_p80': peak_metrics(test_z[z_col], pred_z, thresh_z, timestamps=test_z['timestamp']),
        }
        if timestamps is None:
            timestamps = test_z['timestamp']
            actual_system_test = test_z[TARGET].to_numpy(dtype=float)
            train_system_series = train_z[TARGET]

    # Bottom-up sum of independent zone forecasts
    bottom_up_pred = sum(zone_preds_test.values())
    bottom_up_naive = sum(zone_naive_test.values())

    # Direct combined system forecast on identical window
    df_sys = (
        pd.read_csv(PROCESSED_DATA, parse_dates=['timestamp'])
        if PROCESSED_DATA.exists()
        else build_horizon_table(path=raw_path, target_col=TARGET, horizon=1)
    )
    train_s, valid_s, test_s = temporal_split(df_sys)
    feats_s = feature_columns(df_sys)
    model_s = _make_lgbm('regression_l1')
    model_s.fit(
        train_s[feats_s], train_s[TARGET],
        eval_set=[(valid_s[feats_s], valid_s[TARGET])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    direct_pred = model_s.predict(test_s[feats_s])
    sys_thresh = float(train_s[TARGET].quantile(0.80))

    comparison = {
        'direct_combined_model': {
            **regression_metrics(actual_system_test, direct_pred, bottom_up_naive),
            'peak_metrics_p80': peak_metrics(actual_system_test, direct_pred, sys_thresh, timestamps=timestamps),
        },
        'bottom_up_zone_sum_model': {
            **regression_metrics(actual_system_test, bottom_up_pred, bottom_up_naive),
            'peak_metrics_p80': peak_metrics(actual_system_test, bottom_up_pred, sys_thresh, timestamps=timestamps),
        },
        'zones': zone_reports,
    }
    ZONE_METRICS.write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    return comparison


def fit_baseline(data_path: Path | str = PROCESSED_DATA, run_extended_studies: bool = True):
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    train, valid, test = temporal_split(df)
    features = feature_columns(df)

    # 1. Point LightGBM baseline (L1 objective)
    model = _make_lgbm(objective='regression_l1')
    model.fit(
        train[features], train[TARGET],
        eval_set=[(valid[features], valid[TARGET])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )

    # 2. Regularized linear regression benchmark (Ridge + StandardScaler)
    ridge_model = fit_ridge_benchmark(train, valid, features, TARGET)

    # 3. Probabilistic Quantile LightGBM (P10, P50, P90) + Validation CQR calibration
    q_models, cqr_pad = fit_quantile_models(train, valid, features, TARGET)
    q_preds = predict_quantiles(q_models, test, features, cqr_pad=cqr_pad)

    test = test.copy()
    test['prediction_lgbm'] = model.predict(test[features])
    test['prediction_ridge'] = ridge_model.predict(test[features])
    test['prediction_seasonal_naive'] = test['seasonal_naive_168h']
    test['prediction_seasonal_naive_24h'] = test['seasonal_naive_24h']
    test['prediction_lgbm_p10_raw'] = q_preds['p10_raw']
    test['prediction_lgbm_p50'] = q_preds['p50']
    test['prediction_lgbm_p90_raw'] = q_preds['p90_raw']
    test['prediction_lgbm_p10'] = q_preds['p10_cal']
    test['prediction_lgbm_p90'] = q_preds['p90_cal']

    # Capacity scenario: threshold fixed from training only to avoid test leakage.
    capacity_threshold = float(train[TARGET].quantile(0.80))
    test['capacity_threshold'] = capacity_threshold
    test['actual_peak'] = (test[TARGET] >= capacity_threshold).astype(int)
    test['predicted_peak'] = (test['prediction_lgbm'] >= capacity_threshold).astype(int)
    test['predicted_peak_p90'] = (test['prediction_lgbm_p90'] >= capacity_threshold).astype(int)

    # Sensitivity table across P80, P90, P95, P97.5 and reserve margins
    sensitivity_df = evaluate_threshold_sensitivity(
        test_df=test,
        train_load=train[TARGET],
        percentiles=CAPACITY_PERCENTILES,
        reserve_margins=RESERVE_MARGINS,
    )
    SENSITIVITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(SENSITIVITY_CSV, index=False)

    # Explainability (TreeSHAP + Permutation Importance)
    explainability = compute_explainability(model, test, features, TARGET)

    metrics = {
        'split': {
            'train_start': str(train['timestamp'].min()),
            'train_end': str(train['timestamp'].max()),
            'valid_start': str(valid['timestamp'].min()),
            'valid_end': str(valid['timestamp'].max()),
            'test_start': str(test['timestamp'].min()),
            'test_end': str(test['timestamp'].max()),
            'train_rows': len(train), 'valid_rows': len(valid), 'test_rows': len(test),
        },
        'seasonal_naive': regression_metrics(test[TARGET], test['prediction_seasonal_naive']),
        'seasonal_naive_24h': regression_metrics(
            test[TARGET], test['prediction_seasonal_naive_24h'], test['prediction_seasonal_naive']
        ),
        'ridge_regression': regression_metrics(
            test[TARGET], test['prediction_ridge'], test['prediction_seasonal_naive']
        ),
        'lightgbm': regression_metrics(
            test[TARGET], test['prediction_lgbm'], test['prediction_seasonal_naive']
        ),
        'lightgbm_probabilistic_raw': probabilistic_metrics(
            test[TARGET],
            test['prediction_lgbm_p10_raw'],
            test['prediction_lgbm_p50'],
            test['prediction_lgbm_p90_raw'],
        ),
        'lightgbm_probabilistic_calibrated': {
            **probabilistic_metrics(
                test[TARGET],
                test['prediction_lgbm_p10'],
                test['prediction_lgbm_p50'],
                test['prediction_lgbm_p90'],
            ),
            'validation_cqr_padding': cqr_pad,
        },
        'peak_capacity_scenario': peak_metrics(
            test[TARGET], test['prediction_lgbm'], capacity_threshold, timestamps=test['timestamp']
        ),
        'peak_capacity_scenario_p90_policy': peak_metrics(
            test[TARGET], test['prediction_lgbm_p90'], capacity_threshold, timestamps=test['timestamp']
        ),
        'threshold_percentiles_train': {
            f'P{p*100:g}': float(train[TARGET].quantile(p)) for p in CAPACITY_PERCENTILES
        },
        'explainability_summary': explainability,
        'best_iteration': int(model.best_iteration_ or model.n_estimators),
        'n_features': len(features),
        'features': features,
    }

    if run_extended_studies:
        backtest_summary = run_rolling_backtest(df, n_folds=5, fold_days=30)
        horizon_summary = run_multi_horizon_study()
        zone_summary = run_zone_comparison_study()
        metrics['rolling_backtest_aggregate'] = backtest_summary['aggregate']
        metrics['multi_horizon_summary'] = {
            k: {
                'lightgbm_wape': v['lightgbm']['wape'],
                'lightgbm_mase': v['lightgbm']['mase_vs_seasonal_naive'],
                'empirical_coverage_80': v['probabilistic']['empirical_coverage'],
                'mean_pinball_loss': v['probabilistic']['mean_pinball_loss'],
                'peak_recall_p80': v['peak_metrics']['peak_recall'],
                'mean_abs_peak_timing_error_hours': v['peak_metrics']['mean_abs_peak_timing_error_hours'],
            }
            for k, v in horizon_summary.items()
        }
        metrics['zone_comparison_summary'] = {
            'direct_combined_wape': zone_summary['direct_combined_model']['wape'],
            'bottom_up_sum_wape': zone_summary['bottom_up_zone_sum_model']['wape'],
            'zone_wapes': {z: d['lightgbm']['wape'] for z, d in zone_summary['zones'].items()},
        }

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        'timestamp', TARGET,
        'prediction_lgbm', 'prediction_ridge',
        'prediction_seasonal_naive', 'prediction_seasonal_naive_24h',
        'prediction_lgbm_p10', 'prediction_lgbm_p50', 'prediction_lgbm_p90',
        'prediction_lgbm_p10_raw', 'prediction_lgbm_p90_raw',
        'capacity_threshold', 'actual_peak', 'predicted_peak', 'predicted_peak_p90',
    ]
    test[keep].to_csv(PREDICTIONS, index=False)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # LightGBM's native file writer can fail on Windows when the project path
    # contains non-ASCII characters. Serialize the model in Python and let
    # pathlib perform the Unicode-safe write instead.
    MODEL_PATH.write_text(model.booster_.model_to_string(), encoding='utf-8')
    MODEL_P10_PATH.write_text(q_models[0.10].booster_.model_to_string(), encoding='utf-8')
    MODEL_P50_PATH.write_text(q_models[0.50].booster_.model_to_string(), encoding='utf-8')
    MODEL_P90_PATH.write_text(q_models[0.90].booster_.model_to_string(), encoding='utf-8')
    return model, metrics, test[keep]


if __name__ == '__main__':
    model, metrics, predictions = fit_baseline()
    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {PREDICTIONS}")
