# Executive brief: Tetouan hourly load forecasting

## Recommendation

Treat the model as a **promising analysis prototype**, not an operational capacity recommendation. The LightGBM model performed better than the weekly seasonal-naive benchmark on the available December 2017 holdout, so the next step is a shadow evaluation using current, unit-documented data and an operator-approved capacity threshold. Do not use the slider's default threshold as a real grid limit.

## Decision and evidence

The analysis asks whether an operator could use a one-hour-ahead forecast of combined hourly load to prepare for high-load periods. On the 720 hourly observations from 1–30 December 2017:

- LightGBM MAE was **1,320** versus **1,775** for seasonal naive, an observed reduction of about **455 load units per hour (25.6%)**.
- WAPE was **2.07%** versus **2.78%**; MASE versus seasonal naive was **0.744**.
- At the training-set 80th-percentile scenario threshold (**about 90,061 source units**), the model anticipated **13 of 15** actual threshold breaches (recall **86.7%**), with **9 false alerts** among 22 forecast alerts (precision **59.1%**).

These are retrospective holdout results, not a guaranteed future performance range. The threshold and reserve-margin sensitivity views show how alert counts and illustrative reserve quantities change under alternative assumptions; they do not estimate financial savings or avoided outages.

## Expected impact and limits

The measured benefit is lower forecast error on one historical month. The source does not establish physical units or an actual grid capacity, and the project has no intervention costs, reserve costs, or outage outcomes. A monetary impact range therefore cannot be responsibly estimated from this analysis.

Key risks are the single-year dataset, one fixed test period, possible changes in demand patterns since 2017, and the absence of a real weather forecast at decision time. Point forecasts also do not quantify uncertainty. The model should not drive dispatch or investment decisions until rolling-origin backtests, a calibrated prediction interval, unit validation, and a business-approved threshold are available.

## Suggested next step

Run the model in shadow mode on recent data. Compare it with the existing planning method across multiple chronological folds; have operations choose a meaningful capacity threshold and alert tolerance; then assess false-alert and missed-breach costs before considering a controlled pilot.
