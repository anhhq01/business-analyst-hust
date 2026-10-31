from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / 'data' / 'raw' / 'Tetuan City power consumption.csv'
PROCESSED_DATA = ROOT / 'data' / 'processed' / 'hourly_features.csv'
DQ_SCORECARD_JSON = ROOT / 'data' / 'processed' / 'dq_scorecard.json'
PREDICTIONS = ROOT / 'reports' / 'predictions.csv'
METRICS = ROOT / 'reports' / 'metrics.json'
BACKTEST_METRICS = ROOT / 'reports' / 'backtest_metrics.json'
BACKTEST_PREDICTIONS = ROOT / 'reports' / 'backtest_predictions.csv'
SENSITIVITY_CSV = ROOT / 'reports' / 'threshold_sensitivity.csv'
HORIZON_METRICS = ROOT / 'reports' / 'multi_horizon_metrics.json'
ZONE_METRICS = ROOT / 'reports' / 'zone_comparison_metrics.json'
SHAP_SUMMARY_CSV = ROOT / 'reports' / 'shap_importance.csv'
SHAP_LOCAL_JSON = ROOT / 'reports' / 'shap_peak_explanations.json'
MODEL_PATH = ROOT / 'models' / 'lgbm_1h.txt'
MODEL_P10_PATH = ROOT / 'models' / 'lgbm_1h_p10.txt'
MODEL_P50_PATH = ROOT / 'models' / 'lgbm_1h_p50.txt'
MODEL_P90_PATH = ROOT / 'models' / 'lgbm_1h_p90.txt'
EVIDENTLY_HTML = ROOT / 'reports' / 'evidently_regression.html'
EVIDENTLY_WORKSPACE = Path(tempfile.gettempdir()) / 'Tetouan_Evidently_UI'
EVIDENTLY_PROJECT_NAME = 'Tetouan hourly load evaluation'

TIMESTAMP_COL = 'DateTime'
ZONE_COLS = [
    'Zone 1 Power Consumption',
    'Zone 2 Power Consumption',
    'Zone 3 Power Consumption',
]
WEATHER_COLS = [
    'Temperature',
    'Humidity',
    'Wind Speed',
    'general diffuse flows',
    'diffuse flows',
]
TARGET = 'system_load'
FORECAST_HORIZON_HOURS = 1
MULTI_HORIZONS = [1, 6, 24]
QUANTILES = [0.10, 0.50, 0.90]
CAPACITY_PERCENTILES = [0.80, 0.90, 0.95, 0.975]
RESERVE_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]
SEASONAL_PERIOD_HOURS = 168  # same hour one week ago
TEST_DAYS = 30
VALID_DAYS = 30
RANDOM_STATE = 42
