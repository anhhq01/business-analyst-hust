# Data Dictionary

## 1. Raw Telemetry Columns (`data/raw/Tetuan City power consumption.csv`)

| Column Name | Role | Data Type | Observed Range (Hourly Mean) | Preprocessing & Leakage Treatment |
| :--- | :--- | :--- | :---: | :--- |
| `DateTime` | Timestamp index | `datetime64[ns]` | `2017-01-01 00:00` – `2017-12-30 23:00` | Parsed, sorted, checked for 10-min continuity; aggregated to hourly index `timestamp`. |
| `Temperature` | Weather covariate | `float64` | `3.60` – `39.78` (°C) | Hourly mean; **excluded at target hour $T$**. Only lagged ($T-h, T-24$) and shifted rolling means enter models. |
| `Humidity` | Weather covariate | `float64` | `13.20` – `94.77` (%) | Hourly mean; **excluded at $T$**. Only lagged and shifted rolling means enter models. |
| `Wind Speed` | Weather covariate | `float64` | `0.05` – `6.45` (m/s) | Hourly mean; **excluded at $T$**. Only lagged and shifted rolling means enter models. |
| `general diffuse flows` | Solar radiation covariate | `float64` | `0.03` – `1,089.83` | Hourly mean; **excluded at $T$**. Only lagged and shifted rolling means enter models. |
| `diffuse flows` | Solar radiation covariate | `float64` | `0.02` – `888.85` | Hourly mean; **excluded at $T$**. Only lagged and shifted rolling means enter models. |
| `Zone 1 Power Consumption` | Zonal load component | `float64` | `14,686` – `51,260` | Hourly mean (`45.5%` of test system load). Contemporaneous value excluded from predictors. |
| `Zone 2 Power Consumption` | Zonal load component | `float64` | `9,342` – `36,690` | Hourly mean (`37.1%` of test system load). Contemporaneous value excluded from predictors. |
| `Zone 3 Power Consumption` | Zonal load component | `float64` | `6,389` – `47,209` | Hourly mean (`17.3%` of test system load). Contemporaneous value excluded from predictors. |

---

## 2. Derived Target & Quality Indicators (`data/processed/hourly_features.csv`)

| Column Name | Role | Formula / Definition | Notes |
| :--- | :--- | :--- | :--- |
| `system_load` | **Primary Target ($y_T$)** | `Zone 1 + Zone 2 + Zone 3` (hourly mean) | Combined city load index (`36,570` – `133,619`). Not labeled as kWh energy because source units are unverified. |
| `observations_per_hour` | Quality audit | Count of 10-min records in hour $T$ | Must equal `6` for complete hours; excluded from model features. |
| `is_complete_hour` | Quality filter | `1` if `observations_per_hour == 6` else `0` | Only complete hours (`1`) are retained; excluded from model features. |
| `seasonal_naive_168h` | Primary Benchmark | `system_load` at $T - 168\text{h}$ | Same hour one week earlier. |
| `seasonal_naive_24h` | Secondary Benchmark | `system_load` at $T - \max(24, h)\text{h}$ | Same hour one day earlier. |

---

## 3. Engineered Feature Groups (63 Leakage-Safe Features for $h=1$)

For any forecast horizon $h \in \{1, 6, 24\}$ hours ahead, all historical load and weather features are shifted by at least $h$ hours ($T - h$ or earlier).

| Feature Group | Columns (`h = 1` setup) | Count | Availability at Decision Time $T - h$ |
| :--- | :--- | :---: | :--- |
| **Deterministic Calendar** | `hour`, `dayofweek`, `month`, `dayofyear`, `is_weekend` | 5 | Known exactly in advance for any target timestamp $T$. |
| **Cyclical Time Encodings** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos` | 6 | $\sin(2\pi t/P), \cos(2\pi t/P)$ for $P \in \{24, 7, 365.25\}$; eliminates midnight/weekend wrap-around jumps. |
| **Historical Load Lags** | `load_lag_1h`, `2h`, `3h`, `6h`, `12h`, `24h`, `25h`, `48h`, `72h`, `168h` | 10 | Strictly historical ($\text{lag} \ge h$). Captures immediate autoregression, diurnal (`24h`), and weekly (`168h`) cycles. |
| **Rolling Load Summaries** | `load_roll_{mean,std,max}_{3,6,12,24,168}h` | 15 | Computed on `system_load.shift(h)` so $y_T$ never enters its own window. |
| **Load Velocity & Differencing** | `load_ramp_1h` ($y_{T-h} - y_{T-h-1}$), `load_diff_24h` ($y_{T-h} - y_{T-h-24}$) | 2 | Captures evening ramp-up speed and day-over-day regime shifts (`BA3`). |
| **Lagged & Rolling Weather** | `{temperature,humidity,wind_speed,general_diffuse_flows,diffuse_flows}_{lag_1h,lag_24h,roll_mean_6h,roll_mean_24h}` | 20 | Strictly shifted by $\ge h$ hours. |
| **Domain Weather Interactions** | `cooling_degree_proxy` ($\max(0, \text{Temp}_{T-h}-22)$), `heating_degree_proxy` ($\max(0, 15-\text{Temp}_{T-h})$), `temp_x_hour_sin`, `temp_x_hour_cos`, `temp_x_humidity` | 5 | Captures U-shaped thermal response and evening cooling/heating interaction (`BA6`). |
| **Total Model Predictors** | All columns not in `EXCLUDE` | **63** | Verified by automated leakage unit tests in [`tests/test_preprocessing.py`](../tests/test_preprocessing.py). |
