# Source Log & Data Quality Scorecard

## 1. Dataset Provenance & Access Metadata

| Field | Details |
| :--- | :--- |
| **Dataset Title** | Power Consumption of Tetouan City |
| **Repository / Host** | [UCI Machine Learning Repository, Dataset 849](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) |
| **DOI** | `10.24432/C5B034` |
| **Creators** | Abdulwahed Salam; Abdelaaziz El Hibaoui (Abdelmalek Essaâdi University, Tetouan, Morocco) |
| **Citation** | Salam, A. & El Hibaoui, A. (2018). *Power Consumption of Tetouan City* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B034 |
| **License / Terms** | Creative Commons Attribution 4.0 International (**CC BY 4.0**) |
| **Verification / Access Date** | Checked October 30–31, 2026 (Original CSV download date not recorded; local workspace copy timestamped 2026-10-30) |
| **Supplied Raw File** | [`data/raw/Tetuan City power consumption.csv`](../data/raw/Tetuan City power consumption.csv) |
| **Raw Dimensions** | `52,416` data rows × `9` columns (UCI catalog header lists 52,417 including header row) |
| **Observed Time Span** | `2017-01-01 00:00:00` through `2017-12-30 23:50:00` (364 complete days) |
| **Raw Sampling Cadence** | Strictly 10-minute intervals (`6` observations per hour) |

---

## 2. Filtering, Joining & Cleaning Decisions

1. **Header Normalization:** Raw column names contain inconsistent internal whitespace (e.g., `Zone 1  Power Consumption` with double spaces). [`normalize_columns()`](../src/preprocessing.py) collapses repeated whitespace to single spaces.
2. **Timestamp Parsing & Sorting:** `DateTime` is parsed with `errors='raise'`, sorted chronologically, and checked for duplicates (`0` found) and non-10-minute gaps (`0` found).
3. **No External Joins:** All weather covariates (`Temperature`, `Humidity`, `Wind Speed`, `general diffuse flows`, `diffuse flows`) and three zone load feeds (`Zone 1`, `Zone 2`, `Zone 3`) are co-located in the primary UCI table; no external unverified datasets were merged.
4. **Hourly Aggregation Rule:** Resampled to `1h` intervals using the **arithmetic mean** (`aggregate_hourly()`). Completeness is enforced by requiring `observations_per_hour == 6` (`is_complete_hour == 1`). All `8,736` hours in the 364-day span are 100% complete (`0` incomplete hours).
5. **Warm-Up Trimming:** Because the weekly seasonal lag (`load_lag_168h`) and 168-hour rolling statistics require 7 days of historical warm-up (plus 1 hour shift), the first `168` hourly rows (`2017-01-01` to `2017-01-07`) are dropped rather than back-filled, leaving `8,568` complete, leakage-free modeling rows.
6. **Outlier & Extreme-Value Policy:** Upper-tail hourly load observations (`19` hours above $Q_3 + 1.5\text{IQR} = 124,576$; `35` hours with $|Z| > 3.0$) occur during July–August evening air-conditioning peaks (`18:00–21:00`). Following course guidelines (`BA3`), these are **genuine physical peak demand signals** rather than sensor errors and are intentionally retained without winsorization.

---

## 3. Six-Dimension Data Quality Scorecard (`BA3` Framework)

Generated automatically by [`src/preprocessing.py`](../src/preprocessing.py) and saved to [`data/processed/dq_scorecard.json`](../data/processed/dq_scorecard.json):

| DQ Dimension | Metric Evaluated | Observed Value | SLA / Target Threshold | Status |
| :--- | :--- | :---: | :---: | :---: |
| **1. Completeness** | Missing cell rate across `52,416 × 9` raw cells | `0.00%` (`0` cells) | `< 2.0%` | **PASS** |
| **2. Uniqueness & Consistency** | Duplicate `DateTime` timestamp rate | `0.00%` (`0` rows) | `< 0.5%` | **PASS** |
| **3. Timeliness & Continuity** | Non-10-minute timestamp gaps & incomplete hours | `0` gaps / `0` incomplete hrs | `0` gaps | **PASS** |
| **4. Validity** | Out-of-range physical readings (Temp $\notin [-10, 55]^\circ\text{C}$, Humidity $\notin [0, 100]\%$, Wind $< 0$, Load $\le 0$) | `0` invalid cells | `< 0.1%` | **PASS** |
| **5. Accuracy & Outlier Handling** | Hourly `system_load` $> Q_3 + 1.5\text{IQR}$ (`124,576`) or $|Z| > 3$ | `19` IQR / `35` $|Z|>3$ | Audit & retain true summer peaks | **PASS** |
| **6. Provenance** | Source DOI, URL, download log, and `CC BY 4.0` license verification | `10.24432/C5B034` | Verified & documented | **PASS** |
