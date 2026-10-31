# Browser-Based Demonstration Guide (Streamlit + Evidently)

## 1. Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: regenerate all processed data, models, backtests, and Evidently HTML
python src/run_pipeline.py
# Launch interactive dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Precomputed predictions, calibrated `P10–P90` quantile intervals, 5-fold rolling backtest results, multi-horizon/zone benchmarks, and TreeSHAP explanations are already packaged in `reports/`.

---

## 2. Dashboard Architecture (5 Interactive Tabs)

1. **`1. Forecast & Intervals`:**
   - Displays actual load vs. selected forecast (`LightGBM Point L1`, `LightGBM Median P50`, `LightGBM Risk-Aware P90`, `Standardized Ridge`, `Seasonal Naive 168h / 24h`).
   - Overlays the **validation-calibrated `P10–P90` probabilistic forecast interval (80% CQR)** (or raw quantile interval) and scenario capacity threshold.
   - Compares diurnal hourly profiles and model accuracy/peak metrics side by side.
2. **`2. Capacity & Sensitivity`:**
   - Interactive simulator for capacity threshold presets (`P80`, `P90`, `P95`, `P97.5`, or custom slider), reserve margins (`0%–30%`), and asymmetric shortfall penalty ratios (`2:1` to `25:1`).
   - Displays the full multi-threshold sensitivity matrix (`reports/threshold_sensitivity.csv`) and downloadable hour-by-hour dispatch alerts.
3. **`3. Backtesting, Horizons & Zones`:**
   - **5-Fold Expanding-Window Rolling Backtest** (`Aug–Dec 2017`) showing fold-by-fold stability across summer and winter regimes.
   - **Multi-Horizon Comparison** (`1h` vs. `6h` vs. `24h` ahead) with strict horizon-shifted lags.
   - **Zone-Specific vs. Pooled Comparison & Subgroup Fairness** across Zones 1, 2, and 3.
4. **`4. Explainability (SHAP) & Diagnostics`:**
   - Global **TreeSHAP** feature attribution (`reports/shap_importance.csv`) and **Local Peak-Hour Waterfall/Bar Explanations** (`reports/shap_peak_explanations.json`) for top December peak hours.
   - Residual histogram and Residuals vs. Fitted heteroscedasticity diagnostics (`BA6`).
5. **`5. Data Quality & Evidently`:**
   - Displays the **6-Dimension Data Quality Scorecard** (`data/processed/dq_scorecard.json`) and embeds/downloads the standalone **Evidently Regression Report** (`reports/evidently_regression.html`).

---

## 3. Recommended 5-Minute Live Demo Script

1. **Frame the Decision (30s):** Point to the top KPI ribbon (`WAPE 1.42%`, `Peak Recall 93.3%`, `Peak Timing Error 0.27 h`, `P10–P90 Coverage 87.8%`). Emphasize that we solve an operational reserve-sizing decision during `17:00–21:00` peak ramps.
2. **Show Probabilistic Intervals (1m):** In Tab 1, toggle between `LightGBM Point (L1)` and `LightGBM Risk-Aware Upper (P90)` with the `Calibrated P10–P90` band enabled. Zoom into the late-December evening peaks (`Dec 25–27`).
3. **Simulate Reserve Policy & Sensitivity (1.5m):** Switch to Tab 2. Compare `P80` threshold at `0%` vs. `5%` reserve margin, and show how `LightGBM Point` cuts 10:1 scenario cost by `64.6%` vs. `Seasonal Naive`, while `LightGBM Quantile (P90)` cuts unserved peak volume by `39.5%`.
4. **Prove Robustness Across Folds, Horizons & Zones (1m):** Open Tab 3 to show 5-fold rolling backtest stability (`1.62%` mean WAPE, `81.4%` mean coverage), `1h`/`6h`/`24h` degradation, and why direct combined forecasting beats bottom-up zone summing on system peak recall (`93.3%` vs. `53.3%`).
5. **Explain Peak Drivers & Governance (1m):** Show Tab 4's local TreeSHAP explanation for `2017-12-27 18:00` (driven by `load_lag_24h` and the `+14,644` unit `load_ramp_1h`), then show Tab 5's 6-Dimension DQ Scorecard and Evidently report.
