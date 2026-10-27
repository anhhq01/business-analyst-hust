from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / 'data' / 'raw' / 'Tetuan City power consumption.csv'
PROCESSED_DATA = ROOT / 'data' / 'processed' / 'hourly_features.csv'
PREDICTIONS = ROOT / 'reports' / 'predictions.csv'
METRICS = ROOT / 'reports' / 'metrics.json'
MODEL_PATH = ROOT / 'models' / 'lgbm_1h.txt'
EVIDENTLY_HTML = ROOT / 'reports' / 'evidently_regression.html'

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
SEASONAL_PERIOD_HOURS = 168  # same hour one week ago
TEST_DAYS = 30
VALID_DAYS = 30
RANDOM_STATE = 42
