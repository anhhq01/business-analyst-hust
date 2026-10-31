# Executive Brief: Short-Term Electricity Demand Forecasting & Peak Capacity Planning

**Project:** Business Analytics Capstone #8 — Tetouan City Power Grid (`2017` Hourly Telemetry)  
**Target Audience:** Grid Dispatch & Capacity Planning Operations Management  
**Decision Horizon:** 1-hour-ahead operational reserve dispatch (with 6-hour and 24-hour day-ahead planning benchmarks)

---

## 1. Business Decision & Core Recommendation (DOC Framework)

- **Operational Decision (D):** How much standby generation or feeder reserve should grid operators commit ahead of each operating hour—especially during evening peak windows (`17:00–21:00`)—to prevent capacity threshold breaches without over-committing costly spinning reserves?
- **Unit of Analysis & Target:** One complete operating hour $T$; combined city demand `system_load` = sum of hourly mean loads across Zone 1 (`45.5%`), Zone 2 (`37.1%`), and Zone 3 (`17.3%`).
- **Candidate Policy Options (O):**
  1. **Baseline Rule (Weekly Seasonal Naive `168h`):** Dispatch based on load from the same hour one week prior.
  2. **Point-Forecast Rule (`LightGBM L1` / `Ridge`):** Trigger peak alerts and size reserve using expected point demand $\hat{y}_{T|T-1}$.
  3. **Recommended Risk-Aware Quantile Policy (`LightGBM P90` during Peak Windows):** Use point forecasts (`P50`/`L1`) for base-load hours (`22:00–16:00`), and switch to the validation-calibrated **90th percentile forecast (`P90`)** or a **5% dynamic reserve buffer** during high-risk ramp hours (`17:00–21:00`).

> **Primary Recommendation:** Adopt the **Direct Combined-System Forecast + Calibrated `P10–P90` Uncertainty Band**. For 1-hour-ahead dispatch, replace the weekly seasonal rule (`WAPE 2.78%`, peak recall `46.7%`) with the leakage-safe ML/regularized pipeline (`LightGBM WAPE 1.42%`, `Ridge WAPE 1.06%`, peak recall `93.3%`). Under an illustrative 10:1 asymmetric penalty ratio (unmet peak load penalty vs. standby reserve cost at the `P80` capacity threshold), `LightGBM Point` reduces total scenario cost by **64.6%** relative to Seasonal Naive (`28,747` vs. `81,141` scenario cost units), while the **Calibrated `P90` Quantile Policy** cuts unserved peak exceedance volume by **39.5%** (`788` vs. `1,302` load units) and reduces daily peak under-prediction frequency from `36.7%` to **`10.0%`**.

---

## 2. Forecast Accuracy, Backtesting Stability & Peak Window Insights

All features strictly use information available by $T - h$ for horizon $h$ (zero target or weather leakage).

### 2.1 Holdout (Dec 2017) & 5-Fold Rolling-Origin Backtest (Aug–Dec 2017)
| Model / Policy | Dec 2017 WAPE | Dec 2017 MASE | 5-Fold Mean WAPE (±SD) | 5-Fold Mean MASE | Peak Recall (`P80`) | Peak Precision (`P80`) | Mean Peak Timing Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (`168h`)** | 2.78% | 1.000 | 4.12% (±1.85%) | 1.000 | 46.7% | 58.3% | 0.53 h |
| **Ridge Regression (Std)** | **1.06%** | **0.380** | **1.55%** (±0.46%) | **0.404** | 86.7% | 76.5% | 0.27 h |
| **LightGBM Point (`L1`)** | 1.42% | 0.509 | 1.62% (±0.44%) | 0.423 | **93.3%** | **82.4%** | **0.27 h** (5-fold: **0.21 h**) |
| **LightGBM Quantile (`P90`)** | 2.28%* | — | — | — | **93.3%** | 35.9% | 0.40 h |

*\*Note: `P90` intentionally forecasts the 90th percentile upper bound to buffer against upward demand shocks.*

### 2.2 Probabilistic Calibration & Multi-Horizon Trade-offs (`1h`, `6h`, `24h`)
- **Uncertainty Calibration:** Raw LightGBM `P10–P90` intervals achieved `76.9%` empirical coverage on the December test set (slightly below the `80.0%` nominal target). Applying **Split-Conformal Quantile Regression (CQR)** calibrated strictly on the November validation split (`+351.1` load units per side) achieved **`87.8%` test coverage** (`mean pinball loss: 301.0`) and **`81.4%` mean coverage across all 5 rolling folds**.
- **Planning Horizons (`1h` vs. `6h` vs. `24h`):**
  - **1h Ahead (Real-Time Balancing):** `WAPE 1.42%`, `MASE 0.509`, `Peak Timing Error 0.27 h` (`96.7%` of daily peaks predicted within $\pm 1$ hour).
  - **6h Ahead (Intra-Day Unit Commitment):** `WAPE 2.11%`, `MASE 0.757`, `80% Coverage 89.2%`, `Peak Recall 60.0%`.
  - **24h Ahead (Day-Ahead Scheduling):** `WAPE 2.30%`, `MASE 0.828`, `80% Coverage 85.7%`, `Peak Recall 73.3%`, `Peak Timing Error 0.37 h`.
  - *Operational Implication:* Day-ahead (`24h`) schedules should use the `P90` upper bound to pre-commit slow-ramping thermal/import capacity, then refine dispatch at `T-1h` using the tighter 1-hour-ahead interval (`mean width 3,478` vs. `6,070` at `24h`).

---

## 3. Capacity Threshold & Reserve-Margin Scenario Analysis

Because the UCI dataset does not publish Tetouan's physical substation ratings or financial tariff schedules, capacity planning is evaluated across explicit training-distribution thresholds (`P80 = 90,061`, `P90 = 95,872`, `P95 = 100,972`, `P97.5 = 109,925`) and reserve margins (`0%–20%`):

| Policy (`P80 = 90,061` Threshold, Dec 2017) | Reserve Margin | Triggered Alerts | Breaches Caught (`TP / 15`) | False Alerts (`FP`) | Unmet Peak Volume (Units) | Scenario Total Cost (10:1 Shortfall Weight) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (`168h`)** | 0% | 12 | 7 (`46.7%`) | 5 | 7,004.2 | 81,141 |
| **Ridge Regression** | 0% | 17 | 13 (`86.7%`) | 4 | 1,712.2 | 33,842 |
| **LightGBM Point (`L1`)** | 0% | 17 | 14 (`93.3%`) | 3 | 1,302.1 | **28,747** |
| **LightGBM Quantile (`P90`)** | 0% | 39 | 14 (`93.3%`) | 25 | **787.8** | 57,176 |
| **LightGBM Point (`L1`)** | 5% | 72 | **15 (`100.0%`)** | 57 | **0.0** | 240,132 |

- **When Shortfall Penalties Dominate (>15:1 or Critical Grid Reliability):** Dispatching to `LightGBM Quantile (P90)` (0% extra margin) or `LightGBM Point + 5% margin` during evening peak hours (`17:00–21:00`) eliminates or nearly eliminates unserved peak energy without holding a blanket 10%–20% reserve across all 24 hours.
- **Direct Combined vs. Bottom-Up Zone Modeling:** Forecasting combined `system_load` directly outperforms summing independent Zone 1–3 models on both overall accuracy (`WAPE 1.42%` vs. `1.49%`) and system peak recall (`93.3%` vs. `53.3%`), because independent zone models miss coincident cross-zone evening ramps (`18:00`). However, zone-specific models remain valuable for local feeder monitoring, especially in **Zone 3** where relative volatility is highest (`WAPE 3.48%` vs. `1.76%` in Zone 1).

---

## 4. Assumptions, Risks & Scope Guardrails

1. **Non-Causal Scenario Estimates:** All reserve-margin and cost numbers are scenario simulations under transparent parameter assumptions, **not** observed causal savings or physical grid guarantees.
2. **Unverified Source Load Units:** Raw power figures are aggregated via hourly mean (not summed) because the UCI repository does not specify whether 10-minute telemetry represents instantaneous kW/MW or integrated kWh.
3. **Weather Forecast Availability:** To prevent temporal leakage, actual weather at hour $T$ is excluded; only lagged weather ($T-h$) enters the models. Integrating an operational numerical weather prediction (NWP) feed at decision time would further improve `6h` and `24h` summer cooling peak accuracy.
4. **Single-Year Coverage:** The dataset spans a single calendar year (`2017-01-01` to `2017-12-30`), preventing multi-year holiday or multi-year climate drift validation.
