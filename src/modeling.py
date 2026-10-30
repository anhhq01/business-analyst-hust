from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import lightgbm as lgb

from config import (
    PROCESSED_DATA, PREDICTIONS, METRICS, MODEL_PATH, TARGET,
    TEST_DAYS, VALID_DAYS, RANDOM_STATE, WEATHER_COLS, ZONE_COLS,
)
from evaluation import regression_metrics, peak_metrics

EXCLUDE = {
    'timestamp', TARGET, 'seasonal_naive_168h',
    'observations_per_hour', 'is_complete_hour',
    *WEATHER_COLS, *ZONE_COLS,
}


def temporal_split(df: pd.DataFrame):
    df = df.sort_values('timestamp').copy()
    test_start = df['timestamp'].max() - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
    valid_start = test_start - pd.Timedelta(days=VALID_DAYS)
    train = df[df['timestamp'] < valid_start].copy()
    valid = df[(df['timestamp'] >= valid_start) & (df['timestamp'] < test_start)].copy()
    test = df[df['timestamp'] >= test_start].copy()
    return train, valid, test


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def fit_baseline(data_path: Path | str = PROCESSED_DATA):
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    train, valid, test = temporal_split(df)
    features = feature_columns(df)

    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        train[features], train[TARGET],
        # eval_set=[(valid[features], valid[TARGET])],
        eval_X=valid[features],
        eval_y=valid[TARGET],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )

    test = test.copy()
    test['prediction_lgbm'] = model.predict(test[features])
    test['prediction_seasonal_naive'] = test['seasonal_naive_168h']

    # Capacity scenario: threshold fixed from training only to avoid test leakage.
    # 80th percentile is a transparent baseline scenario, not a physical system limit.
    capacity_threshold = float(train[TARGET].quantile(0.80))
    test['capacity_threshold'] = capacity_threshold
    test['actual_peak'] = (test[TARGET] >= capacity_threshold).astype(int)
    test['predicted_peak'] = (test['prediction_lgbm'] >= capacity_threshold).astype(int)

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
        'lightgbm': regression_metrics(
            test[TARGET], test['prediction_lgbm'], test['prediction_seasonal_naive']
        ),
        'peak_capacity_scenario': peak_metrics(
            test[TARGET], test['prediction_lgbm'], capacity_threshold
        ),
        'best_iteration': int(model.best_iteration_ or model.n_estimators),
        'n_features': len(features),
        'features': features,
    }

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        'timestamp', TARGET, 'prediction_lgbm', 'prediction_seasonal_naive',
        'capacity_threshold', 'actual_peak', 'predicted_peak',
    ]
    test[keep].to_csv(PREDICTIONS, index=False)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # LightGBM's native file writer can fail on Windows when the project path
    # contains non-ASCII characters. Serialize the model in Python and let
    # pathlib perform the Unicode-safe write instead.
    MODEL_PATH.write_text(model.booster_.model_to_string(), encoding='utf-8')
    return model, metrics, test[keep]


if __name__ == '__main__':
    model, metrics, predictions = fit_baseline()
    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {PREDICTIONS}")
