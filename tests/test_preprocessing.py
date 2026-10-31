import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from preprocessing import build_model_table, build_horizon_table
from evaluation import (
    wape, mase, pinball_loss, interval_metrics,
    probabilistic_metrics, daily_peak_metrics, peak_metrics,
    evaluate_threshold_sensitivity,
)
from modeling import temporal_split, feature_columns


def test_features_are_historical():
    df, quality = build_model_table(save=False)
    assert df['load_lag_1h'].notna().all()
    assert df['seasonal_naive_168h'].notna().all()
    assert df['timestamp'].is_monotonic_increasing
    assert quality['bad_timestamps'] == 0
    assert quality['duplicate_timestamps'] == 0


def test_no_missing_after_warmup():
    df, _ = build_model_table(save=False)
    assert int(df.isna().sum().sum()) == 0


def test_multi_horizon_no_leakage():
    for h in (1, 6, 24):
        df_h = build_horizon_table(horizon=h)
        feats = feature_columns(df_h)
        # Ensure no raw target or contemporaneous weather columns enter features
        assert 'system_load' not in feats
        assert 'Temperature' not in feats
        assert 'Zone 1 Power Consumption' not in feats
        # Ensure lag columns strictly respect the horizon
        lag_cols = [c for c in feats if '_lag_' in c]
        for c in lag_cols:
            lag_val = int(c.split('_lag_')[1].replace('h', ''))
            assert lag_val >= h, f"Feature {c} violates horizon {h}h"


def test_temporal_split_chronological_and_disjoint():
    df, _ = build_model_table(save=False)
    train, valid, test = temporal_split(df)
    assert train['timestamp'].max() < valid['timestamp'].min()
    assert valid['timestamp'].max() < test['timestamp'].min()
    assert len(test) == 30 * 24
    assert len(valid) == 30 * 24


def test_probabilistic_and_peak_metrics():
    ts = pd.date_range('2017-12-01', periods=48, freq='1h')
    y_true = np.linspace(50000, 95000, 48)
    y_p10 = y_true - 3000
    y_p50 = y_true - 200
    y_p90 = y_true + 3000

    pl_50 = pinball_loss(y_true, y_p50, 0.50)
    assert np.isclose(pl_50, 100.0)

    prob = probabilistic_metrics(y_true, y_p10, y_p50, y_p90)
    assert prob['empirical_coverage'] == 1.0
    assert prob['mean_interval_width'] == 6000.0

    pk = peak_metrics(y_true, y_p50, capacity_threshold=85000.0, timestamps=ts)
    assert pk['actual_breaches'] > 0
    assert 0.0 <= pk['peak_recall'] <= 1.0
    assert pk['mean_abs_peak_timing_error_hours'] == 0.0


def test_threshold_sensitivity_table():
    ts = pd.date_range('2017-12-01', periods=48, freq='1h')
    y_true = np.linspace(60000, 100000, 48)
    test_df = pd.DataFrame({
        'timestamp': ts,
        'system_load': y_true,
        'prediction_lgbm': y_true - 500,
        'prediction_seasonal_naive': y_true - 2500,
        'prediction_lgbm_p90': y_true + 1500,
        'prediction_ridge': y_true - 1000,
    })
    train_s = pd.Series(np.linspace(50000, 95000, 200))
    sens = evaluate_threshold_sensitivity(
        test_df, train_s, percentiles=[0.80, 0.90], reserve_margins=[0.0, 0.10]
    )
    assert not sens.empty
    assert {'threshold_percentile', 'policy', 'peak_recall', 'scenario_total_cost_10to1'}.issubset(sens.columns)


def test_streamlit_dashboard_renders_without_error():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert len(at.exception) == 0, f"Dashboard raised exception: {at.exception}"
    assert len(at.tabs) == 5
    assert len(at.metric) >= 10

