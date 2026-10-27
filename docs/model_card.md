# Model / analysis card

## Purpose
Forecast combined Tetouan system load one hour ahead to support a transparent peak-capacity scenario.

## Target and horizon
- Target: `system_load`
- Unit: hourly timestamp
- Horizon: 1 hour ahead

## Baselines
1. Weekly seasonal-naive (168-hour lag).
2. LightGBM L1 regression using historical load, lagged weather, and deterministic calendar features.

## Validation
Chronological blocked validation:
- Train: 2017-01-08 to 2017-10-31
- Validation: 2017-11-01 to 2017-11-30
- Test: 2017-12-01 to 2017-12-30

No random split is used.

## Leakage controls
- Target-hour actual weather is excluded.
- Every rolling load/weather statistic is shifted by at least one hour.
- Calendar features for the forecasted hour are allowed because they are known in advance.
- Capacity threshold is derived from training data only.

## Baseline test results
- Seasonal naive WAPE: ~2.78%
- LightGBM WAPE: ~2.07%
- LightGBM MASE vs seasonal naive: ~0.744
- Capacity scenario threshold: training 80th percentile (~90,061)
- Peak recall: ~86.7%
- Peak precision: ~59.1%

## Assumptions / limitations
- The threshold is a scenario, not a documented physical capacity.
- Dataset metadata does not provide explicit power units in the UCI variable table.
- The pipeline uses historical weather only; operational performance could improve with a genuine weather forecast available at decision time.
- A single year limits year-over-year seasonality validation.
- Point forecasts alone are insufficient for risk-aware planning; quantile forecasts are the next priority.
