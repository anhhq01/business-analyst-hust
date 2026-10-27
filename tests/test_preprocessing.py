import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from preprocessing import build_model_table


def test_features_are_historical():
    df, _ = build_model_table(save=False)
    assert df['load_lag_1h'].notna().all()
    assert df['seasonal_naive_168h'].notna().all()
    assert df['timestamp'].is_monotonic_increasing


def test_no_missing_after_warmup():
    df, _ = build_model_table(save=False)
    assert int(df.isna().sum().sum()) == 0
