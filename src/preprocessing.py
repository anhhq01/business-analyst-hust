"""Leakage-safe preprocessing for Tetouan City electricity forecasting.

Forecast setup
--------------
Each modeling row is indexed by the *target hour* T. Predictors must be available
no later than T-1 for a 1-hour-ahead forecast, except deterministic calendar
features for T (hour/day/month), which are known in advance.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import RAW_DATA, PROCESSED_DATA, TIMESTAMP_COL, ZONE_COLS, WEATHER_COLS, TARGET


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [" ".join(c.strip().split()) for c in out.columns]
    return out


def validate_raw(df: pd.DataFrame) -> dict:
    required = {TIMESTAMP_COL, *ZONE_COLS, *WEATHER_COLS}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    ts = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
    bad_ts = int(ts.isna().sum())
    duplicate_ts = int(ts.duplicated().sum())
    diffs = ts.sort_values().diff().dropna()
    non_10min_gaps = int((diffs != pd.Timedelta(minutes=10)).sum())

    quality = {
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'missing_cells': int(df.isna().sum().sum()),
        'bad_timestamps': bad_ts,
        'duplicate_timestamps': duplicate_ts,
        'non_10min_gaps': non_10min_gaps,
        'start': str(ts.min()),
        'end': str(ts.max()),
    }
    if bad_ts or duplicate_ts:
        raise ValueError(f"Temporal integrity check failed: {quality}")
    return quality


def load_raw(path: Path | str = RAW_DATA) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='raise')
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    quality = validate_raw(df)
    return df, quality


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10-minute measurements to hourly mean load/weather.

    Power columns are treated as demand/load measurements, so hourly mean is used.
    We do *not* sum power readings, because that would not be an energy quantity
    unless units and integration rules are explicitly defined.
    """
    x = df.set_index(TIMESTAMP_COL).copy()
    agg_map = {c: 'mean' for c in WEATHER_COLS + ZONE_COLS}
    hourly = x.resample('1h').agg(agg_map)

    # A complete hour should contain 6 ten-minute observations.
    counts = x[ZONE_COLS[0]].resample('1h').count()
    hourly['observations_per_hour'] = counts
    hourly['is_complete_hour'] = (counts == 6).astype('int8')

    # Combined system load = sum of hourly mean load across three zones.
    hourly[TARGET] = hourly[ZONE_COLS].sum(axis=1)
    return hourly


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out['hour'] = idx.hour
    out['dayofweek'] = idx.dayofweek
    out['month'] = idx.month
    out['dayofyear'] = idx.dayofyear
    out['is_weekend'] = (idx.dayofweek >= 5).astype('int8')

    # Cyclical encodings reduce artificial discontinuities at 23->0 and Sun->Mon.
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['dow_sin'] = np.sin(2 * np.pi * out['dayofweek'] / 7)
    out['dow_cos'] = np.cos(2 * np.pi * out['dayofweek'] / 7)
    out['doy_sin'] = np.sin(2 * np.pi * out['dayofyear'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['dayofyear'] / 365.25)
    return out


def add_leakage_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create predictors for target y_T using information available by T-1."""
    out = add_calendar_features(df)

    # Load lags. All are strictly historical relative to target timestamp T.
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        out[f'load_lag_{lag}h'] = out[TARGET].shift(lag)

    # Rolling load statistics are shifted first so y_T never enters its own features.
    hist_load = out[TARGET].shift(1)
    for window in [3, 6, 12, 24, 168]:
        out[f'load_roll_mean_{window}h'] = hist_load.rolling(window, min_periods=window).mean()
        out[f'load_roll_std_{window}h'] = hist_load.rolling(window, min_periods=window).std()
        out[f'load_roll_max_{window}h'] = hist_load.rolling(window, min_periods=window).max()

    # Actual weather at T would be future information in a strict 1-hour-ahead setup.
    # Therefore use latest observed weather (T-1), plus a few historical summaries.
    for c in WEATHER_COLS:
        safe = c.lower().replace(' ', '_')
        out[f'{safe}_lag_1h'] = out[c].shift(1)
        out[f'{safe}_lag_24h'] = out[c].shift(24)
        out[f'{safe}_roll_mean_6h'] = out[c].shift(1).rolling(6, min_periods=6).mean()
        out[f'{safe}_roll_mean_24h'] = out[c].shift(1).rolling(24, min_periods=24).mean()

    # Strong benchmark column: same target hour one week earlier.
    out['seasonal_naive_168h'] = out[TARGET].shift(168)
    return out


def build_model_table(path: Path | str = RAW_DATA, save: bool = True) -> tuple[pd.DataFrame, dict]:
    raw, quality = load_raw(path)
    hourly = aggregate_hourly(raw)

    # Do not silently impute incomplete hours. Mark and then keep only complete rows.
    incomplete_hours = int((hourly['is_complete_hour'] == 0).sum())
    hourly = hourly[hourly['is_complete_hour'] == 1].copy()

    featured = add_leakage_safe_features(hourly)
    featured.index.name = 'timestamp'

    # Drop warm-up rows caused by the 168h history requirement.
    featured = featured.dropna().reset_index()

    quality.update({
        'hourly_rows_after_aggregation': int(len(hourly)),
        'incomplete_hours': incomplete_hours,
        'model_rows_after_feature_warmup': int(len(featured)),
        'target': TARGET,
        'forecast_horizon': '1 hour ahead',
        'aggregation': 'hourly mean load/weather',
    })

    if save:
        PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
        featured.to_csv(PROCESSED_DATA, index=False)
        (PROCESSED_DATA.parent / 'data_quality_summary.json').write_text(
            json.dumps(quality, indent=2), encoding='utf-8'
        )
    return featured, quality


if __name__ == '__main__':
    data, report = build_model_table()
    print(json.dumps(report, indent=2))
    print(f"Saved: {PROCESSED_DATA} ({len(data):,} rows)")
