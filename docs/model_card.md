# Model & Analysis Card

## 1. Purpose & Decision Context
Forecast Tetouan City's short-term electricity demand (`system_load`) across `1h` (primary operational horizon), `6h` (intra-day), and `24h` (day-ahead) horizons, and provide **calibrated probabilistic prediction intervals (`P10–P90`)** to support risk-aware peak capacity and reserve-margin planning.

## 2. KPI Driver Tree (`BA01` Framework)
- **North Star Business Metric:** Expected Peak Balancing & Reserve Scenario Cost = $\text{Standby Reserve Cost} + \lambda \times \text{Unmet Peak Shortfall Volume}$ (evaluated at $\lambda = 10:1$ baseline and across sensitivity ranges).
- **Primary Drivers:**
  1. *Peak Breach Detection Rate (`Peak Recall` & `Unmet Exceedance Volume`)* $\leftarrow$ driven by upper-tail forecast accuracy (`P90` quantile calibration) and evening ramp features (`load_ramp_1h`, `load_diff_24h`).
  2. *Over-Commitment Cost (`Peak Precision` & `Extra Reserve Units`)* $\leftarrow$ driven by point forecast precision (`WAPE`, `MAE`) and interval sharpness (`mean_interval_width`, `winkler_score`).
  3. *Dispatch Timing Alignment (`Mean Absolute Peak Timing Error` in hours)* $\leftarrow$ driven by diurnal cyclical encodings and `24h`/`168h` lags.
- **Operational Levers:** Choice of dispatch trigger (`Point L1` vs. `Quantile P90`), reserve margin percentage (`0%–20%`), and lead-time horizon (`24h` pre-commitment vs. `1h` balancing).
- **Guardrails:**
  - *Temporal Leakage Guardrail:* Zero predictors from $> T - h$.
  - *Calibration Guardrail:* Out-of-sample `P10–P90` empirical coverage within $80\% \pm 10\%$.
  - *Subgroup Fairness Guardrail:* Monitor zone-level error disparity across Zones 1, 2, and 3.

---

## 3. Model Portfolio & Validation Design

### 3.1 Models Evaluated
1. **Weekly Seasonal Naive (`168h`) & Daily Naive (`24h`):** Non-parametric domain benchmarks.
2. **Standardized Ridge Regression (`RidgeCV` + `StandardScaler` + `TimeSeriesSplit`):** Regularized linear benchmark (`BA6`) controlling multicollinearity across lag/rolling features.
3. **LightGBM Point Forecast (`objective='regression_l1'`):** Gradient-boosted decision trees trained on MAE loss with early stopping on the chronological validation split.
4. **Conformalized Quantile LightGBM (`P10`, `P50`, `P90` + Validation CQR):** Separate quantile-loss LightGBM models (`alpha = 0.10, 0.50, 0.90`) with split-conformal nonconformity padding (`+351.1` units) calibrated strictly on the validation set (`2017-11-01` to `2017-11-30`).

### 3.2 Validation Protocol
- **Primary Chronological Split:**
  - *Train:* `2017-01-08 00:00` to `2017-10-31 23:00` (`7,128` hours)
  - *Validation (Early Stopping & CQR Calibration):* `2017-11-01 00:00` to `2017-11-30 23:00` (`720` hours)
  - *Held-Out Test:* `2017-12-01 00:00` to `2017-12-30 23:00` (`720` hours)
- **5-Fold Expanding-Window Rolling Backtest:** Evaluates five consecutive 30-day test windows from August 2017 (summer cooling peak) through December 2017 (winter lighting/heating peak).

---

## 4. Quantitative Evaluation Summary

### 4.1 Primary Holdout (Dec 2017) & 5-Fold Rolling Backtest (`1h` Ahead)
- **Seasonal Naive (`168h`):** Test `WAPE = 2.78%` (`MAE = 1,775.4`), 5-Fold Mean `WAPE = 4.12%`, Peak Recall = `46.7%`.
- **Standardized Ridge Regression:** Test `WAPE = 1.06%` (`MAE = 674.8`, `MASE = 0.380`), 5-Fold Mean `WAPE = 1.55%`, Peak Recall = `86.7%`.
- **LightGBM Point (`L1`):** Test `WAPE = 1.42%` (`MAE = 903.8`, `RMSE = 1,250.3`, `MASE = 0.509`), 5-Fold Mean `WAPE = 1.62% ± 0.44%` (`MASE = 0.423`), Peak Recall = **`93.3%`** (`14/15` breaches caught), Peak Precision = **`82.4%`**, Daily Peak WAPE = **`0.97%`**, Mean Peak Timing Error = **`0.27 h`** (`96.7%` within $\pm 1\text{h}$).
- **Probabilistic Quantile Forecast (`P10–P90`):**
  - *Raw Quantile LightGBM:* `Mean Pinball Loss = 298.4`, `Empirical Coverage = 76.9%` (`Width = 2,775.6`).
  - *Validation-Calibrated CQR (`P10–P90`):* `Mean Pinball Loss = 301.0`, `Test Empirical Coverage = 87.8%`, **5-Fold Mean Empirical Coverage = `81.4%`** (vs. `80.0%` target).

### 4.2 Feature Attribution (TreeSHAP & Permutation Importance on Test Set)
Saved in [`reports/shap_importance.csv`](../reports/shap_importance.csv) and [`reports/shap_peak_explanations.json`](../reports/shap_peak_explanations.json):
1. `load_lag_24h` (`Mean |SHAP| = 9,853.2`, `Permutation ΔMAE = +9,085.2`)
2. `load_lag_168h` (`Mean |SHAP| = 2,070.6`, `Permutation ΔMAE = +1,727.3`)
3. `load_lag_1h` (`Mean |SHAP| = 1,779.4`, `Permutation ΔMAE = +1,228.8`)
4. `load_diff_24h` (`Mean |SHAP| = 1,137.3`, `Permutation ΔMAE = +1,123.9`)
5. `load_lag_48h` (`Mean |SHAP| = 799.0`, `Permutation ΔMAE = +739.3`)
6. `load_ramp_1h` (`Mean |SHAP| = 245.9`, `Permutation ΔMAE = +179.7`) — critical driver during the `17:00–18:00` evening surge (`+14,500` load unit ramp).

---

## 5. Ethics, Subgroup Fairness & Zone Comparison Audit

Electricity distribution reliability directly affects public welfare across urban neighborhoods. We audited forecast accuracy across Tetouan's three distribution zones ([`reports/zone_comparison_metrics.json`](../reports/zone_comparison_metrics.json)):

| Distribution Zone | Test Load Share | LightGBM WAPE | Seasonal Naive WAPE | MASE | Zone Peak Recall (`P80`) | Peak Timing Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zone 1** (Primary Urban Core) | `45.5%` | `1.76%` | `2.68%` | `0.655` | `92.9%` | `0.13 h` |
| **Zone 2** (Intermediate Residential/Commercial) | `37.1%` | `2.08%` | `3.78%` | `0.551` | `99.0%` | `0.53 h` |
| **Zone 3** (Smaller/Peripheral Distribution Zone) | `17.3%` | `3.48%` | `5.08%` | `0.685` | N/A (0 winter breaches of annual P80) | `0.77 h` |

- **Fairness & Operational Risk Finding:** Zone 3 exhibits nearly **double the relative percentage error (`WAPE 3.48%`)** and higher peak timing error (`0.77 h`) compared to Zone 1 (`1.76%`, `0.13 h`), driven by stronger summer coastal/tourism seasonality (Zone 3 peaks heavily in July–August and drops sharply in December).
- **Mitigation:** Relying solely on a city-wide threshold or uncalibrated point forecast risks under-serving local feeder peaks in Zone 3 during summer months. Operators should monitor zone-specific `P90` quantile bounds alongside the direct combined system model (`WAPE 1.42%`).

---

## 6. Assumptions, Risks & Limitations
- **Scenario Thresholds:** Capacity thresholds (`P80–P97.5`) and reserve costs (`10:1` shortfall penalty) are transparent scenario parameters, not physical transformer limits.
- **Historical Weather Only:** Models exclude actual weather at hour $T$ to prevent leakage. In production, integrating a true weather forecast feed ($\hat{W}_{T|T-h}$) is recommended, particularly for `6h` and `24h` horizons.
- **Unverified Units:** Load values are treated as hourly mean demand indices (`Zone 1 + Zone 2 + Zone 3`), not integrated energy (MWh).
