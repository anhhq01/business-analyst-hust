# Business Analytics Capstone #8 — Short-Term Electricity Demand Forecasting & Peak Capacity Planning

Reproducible analytics pipeline, probabilistic forecasting suite, capacity-planning simulator, and interactive dashboard using the **UCI Power Consumption of Tetouan City** dataset.

---

## 1. Business Framing (CRISP-DM / DOC)

- **Business Question:** How accurately can short-term electricity demand be forecast, and how can probabilistic (`P10`/`P50`/`P90`) forecasts support risk-aware peak-capacity and reserve-margin planning?
- **Operational Decision:** How much standby capacity/reserve should grid operators commit ahead of each operating hour—especially during evening peak windows (`17:00–21:00`)—to avoid threshold breaches without over-committing expensive spinning reserves?
- **Unit of Analysis:** One complete hourly timestamp $T$.
- **Primary Target:** `system_load` = `Zone 1 + Zone 2 + Zone 3` hourly mean demand (`data/processed/hourly_features.csv`).
- **Forecast Horizons:** `1 hour ahead` (primary real-time balancing), plus `6 hours ahead` (intra-day commitment) and `24 hours ahead` (day-ahead scheduling).
- **Success Metrics:**
  - *Point Accuracy:* `MAE`, `RMSE`, `WAPE`, and `MASE` (vs. 168h weekly seasonal naive).
  - *Probabilistic Calibration:* `Pinball Loss` (`P10`, `P50`, `P90`), `Empirical Interval Coverage` (`80%` nominal target), and `Winkler Score`.
  - *Peak & Decision Metrics:* `Peak Recall`, `Peak Precision`, `Daily Peak WAPE` (magnitude error), `Mean Absolute Peak Timing Error` (in hours), `Unmet Peak Exceedance Volume`, and `Scenario Total Cost` across capacity thresholds (`P80`, `P90`, `P95`, `P97.5`) and reserve margins (`0%–20%`).

---

## 2. Repository Structure

```text
business-analyst-hust/
├── app.py                                 # 5-tab interactive Streamlit decision dashboard
├── DEMO_SCENARIO.md                       # Step-by-step live presentation & demo script
├── requirements.txt                       # Locked Python dependencies
├── LICENSE_DATA.md                        # UCI CC BY 4.0 attribution
├── data/
│   ├── raw/Tetuan City power consumption.csv
│   └── processed/
│       ├── hourly_features.csv            # 8,568 rows × 76 columns (63 leakage-safe features)
│       ├── data_quality_summary.json      # Automated ingestion & continuity checks
│       └── dq_scorecard.json              # 6-Dimension Data Quality Scorecard (BA3)
├── docs/
│   ├── executive_brief.md                 # ≤2-page Executive Capacity-Planning Brief
│   ├── data_dictionary.md                 # Full raw & engineered feature dictionary
│   ├── source_log.md                      # Provenance, URL, access date & DQ audit
│   ├── model_card.md                      # KPI tree, validation, TreeSHAP & zone fairness audit
│   └── dashboard_demo.md                  # Dashboard walkthrough & 5-minute demo script
├── models/
│   ├── lgbm_1h.txt                        # LightGBM L1 point forecast model
│   ├── lgbm_1h_p10.txt                    # LightGBM P10 lower quantile model
│   ├── lgbm_1h_p50.txt                    # LightGBM P50 median quantile model
│   └── lgbm_1h_p90.txt                    # LightGBM P90 upper quantile model
├── notebooks/
│   └── capstone_topic8_analysis.ipynb     # End-to-end EDA, backtesting, SHAP & scenario notebook
├── reports/
│   ├── metrics.json                       # Holdout point, probabilistic, peak & summary metrics
│   ├── predictions.csv                    # Test predictions (LGBM, Ridge, Naive, P10/P50/P90)
│   ├── backtest_metrics.json              # 5-fold expanding-window rolling backtest metrics
│   ├── backtest_predictions.csv           # Out-of-fold predictions across Aug–Dec 2017
│   ├── threshold_sensitivity.csv          # P80/P90/P95/P97.5 × reserve margin scenario table
│   ├── multi_horizon_metrics.json         # 1h, 6h, and 24h ahead comparison metrics
│   ├── zone_comparison_metrics.json       # Direct combined vs. bottom-up zone sum & fairness metrics
│   ├── shap_importance.csv                # Global TreeSHAP & permutation importance table
│   ├── shap_peak_explanations.json        # Local TreeSHAP explanations for top peak hours
│   └── evidently_regression.html          # Standalone Evidently regression diagnostics report
├── src/
│   ├── config.py                          # Paths, horizons, quantiles & scenario parameters
│   ├── preprocessing.py                   # DQ scorecard, hourly aggregation & horizon-safe features
│   ├── modeling.py                        # Naive, Ridge, LightGBM L1/Quantile, CQR, backtest & SHAP
│   ├── evaluation.py                      # Point, pinball/interval, peak timing/magnitude & scenarios
│   ├── visualization.py                   # Evidently HTML report builder & Evidently UI publisher
│   └── run_pipeline.py                    # Single-command end-to-end execution script
└── tests/
    └── test_preprocessing.py              # Automated unit & dashboard test suites
```

---

## 3. Reproducible Run Instructions

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline (preprocessing, models, backtesting, sensitivity, SHAP, Evidently)
python src/run_pipeline.py

# 4. Run automated unit & leakage tests
pytest tests/ -v

# 5. Launch the interactive decision dashboard
streamlit run app.py
```

---

## 4. Key Results Summary

### 4.1 1-Hour-Ahead Holdout (Dec 2017) & 5-Fold Rolling Backtest (Aug–Dec 2017)
| Model / Policy | Dec 2017 WAPE | Dec 2017 MASE | 5-Fold Mean WAPE | Peak Recall (`P80`) | Peak Precision (`P80`) | Daily Peak WAPE | Mean Peak Timing Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (`168h`)** | 2.78% | 1.000 | 4.12% | 46.7% | 58.3% | 2.91% | 0.53 h |
| **Standardized Ridge** | **1.06%** | **0.380** | **1.55%** | 86.7% | 76.5% | 1.15% | 0.27 h |
| **LightGBM Point (`L1`)** | 1.42% | 0.509 | 1.62% | **93.3%** | **82.4%** | **0.97%** | **0.27 h** (5-fold: **0.21 h**) |
| **LightGBM Calibrated `P10–P90`** | — | — | — | **93.3%** (`P90`) | 35.9% (`P90`) | 2.31% (`P90`) | `87.8%` test / **`81.4%`** 5-fold coverage |

### 4.2 Multi-Horizon (`1h`, `6h`, `24h`) & Zone Comparison Highlights
- **Multi-Horizon:** LightGBM outperforms the 168h seasonal benchmark across all horizons (`1h WAPE 1.42% / MASE 0.509`; `6h WAPE 2.11% / MASE 0.757`; `24h WAPE 2.30% / MASE 0.828`), maintaining `85.7%–89.2%` empirical coverage on the calibrated `P10–P90` interval.
- **Direct Combined vs. Zone-Specific Models:** Direct `system_load` modeling achieves lower WAPE (`1.42%` vs. `1.49%`) and substantially higher system peak recall (`93.3%` vs. `53.3%`) than summing independent Zone 1–3 models, while zone-specific models highlight higher relative volatility in peripheral **Zone 3** (`WAPE 3.48%` vs. `1.76%` in Zone 1).

---

## 5. Final Submission Checklist Verification ([BA_Capstone_Topics_2026_MSc class_0723 1.pdf](../BA_Capstone_Topics_2026_MSc%20class_0723%201.pdf), p. 11)

- [x] **Business decision, unit of analysis, target/outcome, and evaluation period explicitly defined:** Documented in [`README.md`](README.md) and [`docs/executive_brief.md`](docs/executive_brief.md).
- [x] **Dataset source, download date, license/terms, and filtering/joining choices documented:** Documented in [`docs/source_log.md`](docs/source_log.md), [`docs/data_dictionary.md`](docs/data_dictionary.md), and [`LICENSE_DATA.md`](LICENSE_DATA.md).
- [x] **No target leakage present; all predictors available at stated decision time:** Verified by `add_leakage_safe_features(..., horizon=h)` in [`src/preprocessing.py`](src/preprocessing.py) and tested in [`tests/test_preprocessing.py`](tests/test_preprocessing.py).
- [x] **Baseline complete and compared fairly with advanced models:** Seasonal Naive (`168h` & `24h`), Standardized Ridge, LightGBM Point (`L1`), and Quantile LightGBM (`P10`/`P50`/`P90`) evaluated on identical chronological splits.
- [x] **Validation matches temporal structure:** Single chronological split + 5-fold expanding-window rolling backtest (`reports/backtest_metrics.json`).
- [x] **Technical metrics and business/decision metrics reported:** `MAE`, `RMSE`, `WAPE`, `MASE`, `Pinball Loss`, `Interval Coverage`, `Peak Recall/Precision`, `Peak Magnitude/Timing Error`, `Unmet Peak Volume`, and `Scenario Cost`.
- [x] **Scenario assumptions visible and sensitivity analysis included:** Evaluated across `P80`, `P90`, `P95`, `P97.5` thresholds and `0%–20%` reserve margins (`reports/threshold_sensitivity.csv`).
- [x] **Dashboard supports a concrete decision:** [`app.py`](app.py) provides interactive threshold, reserve margin, and asymmetric cost simulation with calibrated `P10–P90` uncertainty bands.
- [x] **Executive brief states recommendation, expected impact/range, risks, and limitations:** Provided in [`docs/executive_brief.md`](docs/executive_brief.md).
- [x] **Third party can rerun core analysis from README and code:** Single-command `python src/run_pipeline.py` and `pytest tests/ -v`.
