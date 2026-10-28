# Business Analytics Capstone #8 — Tetouan Electricity Demand Forecasting

Short-term electricity demand forecasting and peak-capacity planning using the **UCI Power Consumption of Tetouan City** dataset.

## 1. Business framing

**Decision:** How much short-term capacity/reserve should an operator prepare for the next hour, especially during peak-risk windows?

**Unit of analysis:** one hourly timestamp.

**Primary target:** `system_load` = Zone 1 + Zone 2 + Zone 3 hourly mean load.

**Forecast horizon:** 1 hour ahead.

**Primary technical metrics:** MAE, RMSE, WAPE, and MASE versus a weekly seasonal-naive benchmark.

**Decision metric:** precision/recall for load exceeding a fixed capacity threshold. The baseline uses the **80th percentile of training load** as a transparent scenario threshold; it is not claimed to be Tetouan's real physical capacity.

## 2. Repository structure

```text
ba_capstone8_tetouan/
├── data/
│   ├── raw/Tetuan City power consumption.csv
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── config.py
│   ├── preprocessing.py     # main focus: quality, hourly aggregation, leakage-safe features
│   ├── modeling.py          # seasonal naive + LightGBM baseline
│   ├── evaluation.py        # forecast + peak metrics
│   ├── visualization.py     # Evidently regression report
│   └── run_pipeline.py
├── tests/test_preprocessing.py
├── requirements.txt
└── README.md
```

## 3. Why preprocessing is the core of this project

The raw data are already unusually clean: there are no nulls in the supplied CSV and timestamps arrive every 10 minutes. The real risks are therefore **temporal integrity and leakage**, not generic missing-value imputation.

### 3.1 Data quality checks

`preprocessing.py` checks:
- required columns;
- parseable timestamps;
- duplicate timestamps;
- 10-minute continuity;
- hourly completeness (6 raw rows per hour);
- missing values after feature warm-up.

### 3.2 Hourly aggregation

Weather and zonal power measurements are aggregated using the **hourly mean**. Power values are not blindly summed because the source metadata does not establish that the columns are interval energy quantities. If the project later confirms units and wants hourly energy, replace this with a physically justified integration rule.

### 3.3 Leakage-safe feature engineering

For a target hour `T`, only information available by `T-1` enters the model:
- load lags: 1, 2, 3, 6, 12, 24, 48, 72, 168 hours;
- rolling load mean/std/max over 3, 6, 12, 24, 168 hours, always computed after a 1-hour shift;
- weather lags and rolling weather summaries, also shifted by 1 hour;
- calendar features for `T` (hour/day/weekend/month), which are known in advance;
- cyclical encodings for hour, day-of-week, and day-of-year.

The script intentionally does **not** use actual weather at target hour `T`. If you later add a real weather forecast feed, forecast weather for `T` can legitimately be added.

## 4. Modeling baseline

Two baselines are produced:

1. **Seasonal naive:** predict the same hour from 7 days earlier (`168h`).
2. **LightGBM regression:** compact supervised baseline with lag/rolling/calendar/weather-history features.

The final 30 days form the test set and the preceding 30 days form validation. Everything before that is training. This preserves chronology and avoids random train/test leakage.

For the final report, extend this to **rolling-origin backtesting** across several monthly/biweekly folds; the code is deliberately kept simple enough for a live class demo first.

## 5. Evidently visualization

Install dependencies and run:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python src/run_pipeline.py
python src/visualization.py
```

Then open:

```text
reports/evidently_regression.html
```

The report includes regression performance and a compact data summary. For a local monitoring UI, Evidently 0.7 can also launch its self-hosted interface with:

```bash
evidently ui
```

The HTML report is preferable for presentation because it is deterministic and requires no server during the demo.

## 6. Recommended live-demo story

Show only 3–5 decision-relevant KPIs:
- test WAPE;
- MASE vs seasonal naive;
- peak recall;
- peak precision;
- number of actual threshold breaches.

Then show one actionable view: **actual vs forecast load around peak windows with the capacity threshold overlaid**. Explain where the model misses peaks and what reserve-margin rule would be triggered.

## 7. Next upgrades after baseline

Prioritize these, in order:
1. rolling-origin backtesting and fold-level stability;
2. explicit probabilistic forecast (LightGBM quantile at P10/P50/P90);
3. empirical interval coverage and pinball loss;
4. threshold sensitivity (P90/P95/P97.5 training load or a supplied business capacity);
5. zone-specific versus pooled/combined models;
6. SHAP only after validation is stable.

Do not jump to LSTM before the temporal baseline and business decision rule are defensible.

---

## Live browser demo (new)

Run `streamlit run app.py` after `pip install -r requirements.txt`; open **http://localhost:8501**. Uses the included historical predictions (no retraining necessary). Four tabs: Forecast overview, Error diagnostics, Capacity decisions, Evidently monitoring. Details and demo script: [`docs/dashboard_demo.md`](docs/dashboard_demo.md).

This is an **interactive historical scenario demo**, not a live grid data feed. Changing the threshold or reserve margin recalculates evaluation/scenario outputs without retraining. The optional residual band is descriptive, not a validated prediction interval.
