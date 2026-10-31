# Live Capstone Presentation & Interactive Demo Scenario Guide
**Course:** Master of Data Science — Business Analytics (HUST)  
**Topic #8:** Short-Term Electricity Demand Forecasting & Peak Capacity Planning (Tetouan City)  
**Dashboard URL:** `http://localhost:8501/`  
**Repository:** [`business-analyst-hust/`](.)

---

## 1. Pre-Demo Launch Checklist (1 Minute Before Presenting)

Open a terminal and start the Streamlit decision dashboard:

```bash
cd "/Users/johnvu/Documents/SOICT/Bussiness Analysis/business-analyst-hust"
source .venv/bin/activate
streamlit run app.py
```

Have these three browser tabs ready to switch between during the presentation:
1. **Tab A (Main Live Demo):** `http://localhost:8501/` (Streamlit 5-Tab Decision Dashboard)
2. **Tab B (Technical Deep-Dive):** [`notebooks/capstone_topic8_analysis.ipynb`](notebooks/capstone_topic8_analysis.ipynb) (Executed EDA, Simpson's Paradox & Diagnostics Notebook)
3. **Tab C (Managerial Handover):** [`docs/executive_brief.md`](docs/executive_brief.md) ($\le 2$-Page Capacity-Planning Executive Brief)

---

## 2. Recommended Team Role Split (3–4 Students, 10–12 Minutes Total)

| Speaker | Role | Time | Primary Focus & Artifacts Shown |
| :--- | :--- | :---: | :--- |
| **Speaker 1** | **Business & Data Governance Lead** | `2.5 min` | CRISP-DM / DOC Problem Framing, KPI Driver Tree, 6-Dimension Data Quality Scorecard (`Tab 5`), and Simpson's Paradox in Weather–Load EDA (`Notebook Sec 3`). |
| **Speaker 2** | **Lead Forecasting Scientist** | `3.0 min` | Leakage-safe feature engineering ($T-h$), Model Benchmarks (`Seasonal Naive`, `Standardized Ridge`, `LightGBM L1`, `Quantile P10/P50/P90 + CQR`), and 5-Fold Rolling Backtesting (`Tab 1` & `Tab 3`). |
| **Speaker 3** | **Operations & Capacity Strategist** | `3.5 min` | Live Scenario Simulation (`Tab 2`): `P80–P97.5` thresholds, `0%–20%` reserve margins, `10:1` asymmetric penalty trade-off, Multi-Horizon (`1h/6h/24h`) dispatch cascade, and Zone 1–3 fairness (`Tab 3`). |
| **Speaker 4** *(or shared)* | **Explainability & Risk Auditor** | `2.0 min` | Global & Local Peak-Hour TreeSHAP (`Tab 4`), Residual Diagnostics, Evidently Monitoring (`Tab 5`), and Scope Guardrails. |

---

## 3. Five Step-by-Step Interactive Demo Scenarios (`http://localhost:8501/`)

### Scenario 1: The Operational Problem & Baseline vs. ML Point Dispatch
**Goal:** Show why the current rule-of-thumb (`Seasonal Naive 168h`) fails during peak windows and quantify the immediate impact of leakage-safe ML forecasting.

- **Dashboard Actions:**
  1. Start on **Tab 1 (`1. Forecast & Intervals`)** with default sidebar settings:
     - `Primary forecast series`: **`LightGBM Point (L1)`**
     - `Evaluation date range`: **`2017/12/01 – 2017/12/30`**
     - `Capacity threshold preset`: **`P80 (Baseline)`** (`90,061.3` units)
     - `Uncertainty band`: **`Calibrated P10–P90 (80% CQR)`**
  2. Point to the **Top 6 KPI Ribbon**:
     - `WAPE`: **`1.42%`** (`MASE 0.509` — nearly 50% lower error than the weekly seasonal benchmark)
     - `Peak Recall`: **`93.3%`** (`14/15` actual capacity breaches caught)
     - `Peak Precision`: **`82.4%`** (only `3` false alarms across 720 operating hours)
     - `Peak Timing Error`: **`0.27 h`** (`97%` of daily peaks predicted within $\pm 1$ hour)
  3. In the sidebar, switch **`Primary forecast series`** from `LightGBM Point (L1)` $\rightarrow$ **`Seasonal Naive (168h)`**, then switch back.
- **What Happens on Screen:**
  - Under `Seasonal Naive (168h)`, `WAPE` jumps to **`2.78%`**, `Peak Recall` collapses from `93.3%` to **`46.7%`** (`7/15` breaches caught, `8` missed breaches!), and `Peak Timing Error` doubles to **`0.53 h`**.
  - Point to the **Bottom-Left Diurnal Profile Chart** on Tab 1: highlight that almost all breach risk is concentrated in the **evening ramp window (`17:00–21:00`)**, where residential lighting and heating/cooling surge simultaneously.
- **Key Talking Point:**
  > *"We did not start with an algorithm; we started with an operational grid decision: how much standby reserve to commit 1 hour ahead during the 17:00–21:00 peak ramp. Relying on last week's load misses more than half of actual capacity breaches (46.7% recall). Our leakage-safe LightGBM and Standardized Ridge models capture 93.3% and 86.7% of peak breaches with an average peak timing error of just 16 minutes (0.27 hours)."*

---

### Scenario 2: Zooming Into the Peak Week & Risk-Aware Probabilistic Dispatch (`P10–P90`)
**Goal:** Demonstrate why point forecasts alone are insufficient for risk-aware capacity planning and how calibrated quantile intervals (`P10–P90`) protect the grid against upward demand shocks.

- **Dashboard Actions:**
  1. On **Tab 1**, use the sidebar **`Evaluation date range`** (or drag-zoom directly on the Plotly chart) to focus on the peak winter week: **`2017/12/23 – 2017/12/30`**.
  2. Toggle **`Uncertainty band on chart`** between **`Raw Quantile P10–P90`** (`76.9%` coverage) and **`Calibrated P10–P90 (80% CQR)`** (`87.8%` test coverage, `81.4%` 5-fold mean coverage).
  3. Switch **`Primary forecast series`** to **`LightGBM Risk-Aware Upper (P90)`** and set **`Reserve margin (%)`** to **`0%`**.
- **What Happens on Screen:**
  - The blue shaded `P10–P90` interval tightly brackets actual demand across all evening spikes between Dec 25 and Dec 27 (`>91,400` load units).
  - Switching to the `P90` upper quantile policy reduces **Unmet Peak Shortfall** (at `0%` extra reserve margin) from `1,302` units (under Point `L1`) to **`788` units (`-39.5%`)**, and cuts the daily peak under-prediction rate from `36.7%` down to **`10.0%`**.
- **Key Talking Point:**
  > *"Raw tree quantiles slightly under-covered the December test set at 76.9%. Using Split-Conformal Quantile Regression (CQR) calibrated strictly on the November validation split—with zero test leakage—we achieved 81.4% mean empirical coverage across our 5 rolling backtest folds and 87.8% in December. During the 17:00–21:00 peak window, dispatching to the P90 upper bound cuts unserved peak energy by nearly 40% without requiring a blanket 24-hour reserve buffer."*

---

### Scenario 3: Interactive Reserve-Margin & Asymmetric Cost Simulation (`Tab 2`)
**Goal:** Satisfy Mandatory Rule #5 and #6 by quantifying business impact under explicit thresholds (`P80`, `P90`, `P95`, `P97.5`), reserve margins (`0%–20%`), and asymmetric cost assumptions.

- **Dashboard Actions:**
  1. Click **Tab 2 (`2. Capacity & Sensitivity`)**.
  2. Set sidebar controls to:
     - `Capacity threshold preset`: **`P80 (Baseline)`** (`90,061.3`)
     - `Reserve margin (%)`: Drag from **`0%`** $\rightarrow$ **`5%`** $\rightarrow$ **`10%`**
     - `Asymmetric penalty ratio (Unmet Peak : Reserve Cost)`: Drag from **`10`** $\rightarrow$ **`20`**
  3. Inspect the **Multi-Threshold & Reserve Policy Sensitivity Table** (`P80`, `P90`, `P95`, `P97.5`) and click the expander **`View hour-by-hour alerts and reserve dispatch schedule`**.
- **What Happens on Screen (Key Numbers to Cite):**
  - **At `0%` Reserve Margin (`10:1` penalty weight, `P80` threshold):**
    - `Seasonal Naive (168h)`: `7,004.2` unmet peak units $\rightarrow$ **Total Scenario Cost = `81,141`**
    - `Ridge Regression`: `1,712.2` unmet peak units $\rightarrow$ **Total Scenario Cost = `33,842`**
    - `LightGBM Point (L1)`: `1,302.1` unmet peak units $\rightarrow$ **Total Scenario Cost = `28,747` (`-64.6%` vs. Naive)**
    - `LightGBM Quantile (P90)`: **`787.8` unmet peak units (`-39.5%` shortfall vs. Point)** $\rightarrow$ **Total Scenario Cost = `57,176`**
  - **At `5%` Reserve Margin (`LightGBM Point`):**
    - `Peak Recall` reaches **`100.0%` (`15/15` breaches caught)** and **`Unmet Peak Units` drops to `0.0`**, but `False Alerts` rise from `3` to `57` (`Total Scenario Cost = 240,132` if held across all hours).
- **Key Talking Point:**
  > *"Because the UCI dataset does not include physical substation limits or financial penalty tariffs, we built a transparent decision simulator across four training percentiles (P80 to P97.5) and reserve margins from 0% to 20%. At a 10:1 shortfall-to-reserve cost ratio, LightGBM Point at 0% margin minimizes total scenario cost (reducing cost by 64.6% vs. Seasonal Naive). If grid reliability requires zero unserved energy, applying a targeted 5% reserve margin or switching to the P90 quantile specifically during the 17:00–21:00 window achieves 93%–100% breach capture while avoiding 24-hour over-commitment."*

---

### Scenario 4: Rolling Backtesting, Multi-Horizon Cascade (`24h -> 6h -> 1h`) & Zone Fairness (`Tab 3`)
**Goal:** Prove temporal stability across seasons, show how `24h`, `6h`, and `1h` forecasts work together in grid operations, and address subgroup fairness across Tetouan's 3 zones.

- **Dashboard Actions:**
  1. Click **Tab 3 (`3. Backtesting, Horizons & Zones`)**.
  2. Walk through the three tables from top to bottom:
     - **Table 1:** `5-Fold Expanding-Window Rolling Backtest (Aug–Dec 2017)`
     - **Table 2:** `Multi-Horizon Forecasting (1h vs 6h vs 24h Ahead)`
     - **Table 3:** `Zone-Specific vs. Combined Model & Subgroup Fairness`
- **What Happens on Screen (Key Numbers to Cite):**
  - **5-Fold Rolling Backtest (`Aug–Dec 2017`):** Across summer air-conditioning peaks (Fold 1: Aug `WAPE 2.20%`, `Peak Recall 99.0%`) through autumn and winter peaks (Fold 5: Dec `WAPE 1.21%`), LightGBM achieves **`1.62% ± 0.44%` mean WAPE** (`MASE 0.423`), **`81.4%` mean interval coverage** (target `80.0%`), and **`0.21 h` (~13 min) mean peak timing error**.
  - **Multi-Horizon Cascade (`24h -> 6h -> 1h`):**
    - `24h Ahead (Day-Ahead Unit Commitment)`: `WAPE 2.30%`, `MASE 0.828`, `80% Coverage 85.7%`, `Peak Recall 73.3%`.
    - `6h Ahead (Intra-Day Peaker Warm-Up)`: `WAPE 2.11%`, `MASE 0.757`, `80% Coverage 89.2%`.
    - `1h Ahead (Real-Time Dispatch Balancing)`: `WAPE 1.42%`, `MASE 0.509`, `80% Coverage 87.8%`, `Peak Recall 93.3%`.
  - **Direct Combined vs. Bottom-Up Zone Sum & Subgroup Fairness:**
    - `Direct Combined Model` beats `Bottom-Up Sum of Zones` on both `WAPE` (`1.42%` vs. `1.49%`) and `Peak Recall P80` (**`93.3%` vs. `53.3%`**), because independent zone errors compound during coincident evening ramps.
    - **Subgroup Fairness Finding:** `Zone 1` (`45.5%` load share, `WAPE 1.76%`) and `Zone 2` (`37.1%` share, `WAPE 2.08%`) are highly stable, whereas peripheral **`Zone 3` (`17.3%` share) has nearly double the percentage error (`WAPE 3.48%`) and `0.77 h` peak timing error**.
- **Key Talking Point:**
  > *"For system-wide reserve commitment, our Direct Combined model catches 93.3% of breaches compared to only 53.3% when summing separate zone models. However, our fairness audit reveals that Zone 3 experiences nearly twice the relative forecast error (3.48% vs. 1.76% in Zone 1) due to sharp summer coastal demand swings. Therefore, operators should pair the Direct Combined system model with local P90 feeder monitoring for Zone 3."*

---

### Scenario 5: Opening the Black Box (TreeSHAP) & Data Quality Governance (`Tab 4` & `Tab 5`)
**Goal:** Explain *why* the model predicts a peak breach on a specific evening and demonstrate production-grade data quality and drift monitoring.

- **Dashboard Actions:**
  1. Click **Tab 4 (`4. Explainability (SHAP) & Diagnostics`)**.
  2. Point to the **Global TreeSHAP Bar Chart** and open the expander **`View Global TreeSHAP vs. Permutation Importance (ΔMAE) & Split Gain Table`**.
  3. In the right-hand dropdown **`Inspect local TreeSHAP explanation for top peak hour`**, select **`2017-12-27 18:00:00 (Actual: 91,628)`** (and optionally toggle to `2017-12-25 18:00:00`).
  4. Click **Tab 5 (`5. Data Quality & Evidently`)**, point to the **6-Dimension Data Quality Scorecard**, and check the box **`Embed saved Evidently report`**.
- **What Happens on Screen:**
  - **Global Drivers:** `load_lag_24h` (`Mean |SHAP| = 9,853`, `Permutation ΔMAE = +9,085`), `load_lag_168h` (`2,071`), `load_lag_1h` (`1,779`), `load_diff_24h` (`1,137`), and `load_ramp_1h` (`246`) dominate predictions.
  - **Local Peak Explanation (`2017-12-27 18:00:00`):** Base expected load is `72,693`. The model predicts `92,678` (correctly triggering a `P80` alert for the actual `91,628` peak) because of yesterday's 18:00 peak (`load_lag_24h`: `+13,784`), last week's 18:00 peak (`load_lag_168h`: `+1,780`), the 17:00 load level (`load_lag_1h`: `+1,377`), and the steep 1-hour evening ramp (`load_ramp_1h = +14,644` adding `+616` SHAP units).
  - **Tab 5 Governance:** All 6 DQ dimensions (`Completeness`, `Uniqueness`, `Timeliness`, `Validity`, `Accuracy/Outliers`, `Provenance`) show `PASS`, and the embedded **Evidently HTML Report** provides full residual and drift monitoring.

---

## 4. Instructor Q&A Defense Cheat Sheet (Top 8 Questions & Answers)

1. **Q: "Why did you aggregate 10-minute power readings using the hourly mean instead of summing them?"**
   - **A:** Summing six 10-minute power readings (e.g., in kW or MW) produces $6 \times \text{average power}$, which is neither average power nor hourly energy (kWh/MWh requires integrating $\sum P_i \times \frac{1}{6}\text{h}$). Furthermore, the UCI dataset documentation does not verify the physical unit of the raw columns. Taking the **hourly arithmetic mean** preserves the exact scale of the source telemetry and requires all `6` observations per hour (`is_complete_hour == 1`).

2. **Q: "How did you guarantee zero target leakage and zero weather leakage across `1h`, `6h`, and `24h` horizons?"**
   - **A:** In [`src/preprocessing.py`](src/preprocessing.py), every load lag, rolling window (`3h–168h`), ramp (`load_ramp_1h`), and weather covariate (`Temperature`, `Humidity`, `Wind Speed`, `diffuse flows`) is explicitly shifted by at least the forecast horizon $h$ (`shift(h)` before computing rolling windows). Actual weather at target hour $T$ is **never** used. Automated unit tests in [`tests/test_preprocessing.py`](tests/test_preprocessing.py) verify that `min_lag >= h` for every predictor.

3. **Q: "Why does Standardized Ridge Regression achieve slightly lower WAPE (`1.06%`) than LightGBM (`1.42%`) on the 1h December holdout, yet you recommend LightGBM + Quantile LightGBM for capacity planning?"**
   - **A:** At a 1-hour horizon, electricity load is highly autoregressive (`load_lag_1h`, `load_lag_24h`, `load_lag_168h`), so an L2-regularized linear combination tracks smooth winter diurnal curves exceptionally well (`Ridge WAPE 1.06%`, `Peak Recall 86.7%`). However:
     1. On **peak breach detection (`P80`)**, `LightGBM Point (L1)` achieves higher **Peak Recall (`93.3%` vs. `86.7%`)**, higher **Peak Precision (`82.4%` vs. `76.5%`)**, lower **Unmet Peak Volume (`1,302` vs. `1,712` units)**, and **15% lower 10:1 scenario cost (`28,747` vs. `33,842`)**.
     2. At longer horizons (**`6h` and `24h` ahead**), where immediate `t-1h` autoregression is unavailable and nonlinear calendar-weather interactions matter more, **LightGBM decisively beats Ridge** (`6h WAPE 2.11%` vs. `2.42%`; `24h WAPE 2.30%` vs. `2.60%`).
     3. LightGBM natively supports non-parametric **Quantile Regression (`P10`/`P50`/`P90`)** for asymmetric risk planning.

4. **Q: "Why didn't you remove or winsorize the 35 hourly load outliers ($|Z| > 3$) during data cleaning?"**
   - **A:** Per the `BA3` Data Quality lecture (*"Note: outliers may carry business signal"*), we audited all `19` IQR-high (`> 124,576`) and `35` $|Z| > 3$ hours. Every single one occurred during July–August evening cooling peaks (`18:00–21:00`) when temperatures exceeded `30°C`—with zero sensor dropouts or negative spikes. Winsorizing them would blind the model to the exact peak-capacity events our project is designed to anticipate.

5. **Q: "Where does Simpson's Paradox appear in this dataset?"**
   - **A:** Shown in Section 3 of [`notebooks/capstone_topic8_analysis.ipynb`](notebooks/capstone_topic8_analysis.ipynb): pooling all `8,568` hours gives a positive correlation between Temperature and Load (`r = +0.369`), and even pooling all Winter hours (`Dec–Feb`) gives `r = +0.206` because daytime hours are simultaneously warmer and have higher commercial activity than 04:00 AM (**hour-of-day confounding**). Once we stratify *within* each operating hour in Winter, the correlation **reverses sign to negative** (e.g., `r = -0.351` at the `19:00` winter evening peak) because colder winter evenings drive electric heating load.

6. **Q: "How did you calibrate your `P10–P90` prediction intervals without leaking test data?"**
   - **A:** In [`src/modeling.py`](src/modeling.py) (`fit_quantile_models`), we trained `alpha = 0.10, 0.50, 0.90` LightGBM models on `train` (`Jan–Oct 2017`) and computed split-conformal nonconformity scores $E_i = \max(\hat{q}_{0.10}(x_i) - y_i, y_i - \hat{q}_{0.90}(x_i))$ strictly on the chronological `validation` set (`Nov 2017`). The 80th percentile validation padding (`+351.1` load units) was then frozen and applied to `test` (`Dec 2017`), improving empirical coverage from `76.9%` (raw) to **`87.8%` on Dec test** and **`81.4%` across the 5 rolling folds**.

7. **Q: "Why didn't you use an LSTM or Deep Learning model?"**
   - **A:** Topic 8's Scope Guardrails explicitly state: *"Deep learning is not required. A well-validated seasonal/statistical or gradient-boosting model is preferred to an under-tuned LSTM."* With `8,568` tabular hourly rows and engineered lag/rolling/cyclical features, LightGBM and Ridge achieve `1.06%–1.42%` WAPE, train in seconds across 5 backtest folds, and provide exact TreeSHAP attribution.

8. **Q: "Can you claim that your model saves the Tetouan utility $X%$ in real dollars?"**
   - **A:** **No.** Per Mandatory Analytics Rule #5, observational load telemetry does not identify causal financial savings or physical grid parameters. We report **scenario-based impact estimates** (`-64.6%` scenario cost under a 10:1 shortfall penalty at `P80` and `-39.5%` unmet peak volume under `P90`) with full sensitivity tables across thresholds (`P80–P97.5`), reserve margins (`0%–20%`), and penalty ratios (`2:1–25:1`).
