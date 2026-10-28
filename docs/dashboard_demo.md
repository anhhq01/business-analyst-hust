# Browser-based demonstration (Streamlit + Evidently)

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the browser at `http://localhost:8501`. Precomputed December 2017 predictions are packaged, so you **do not** have to rerun model training to demonstrate the dashboard. For fresh historical retraining: `python src/run_pipeline.py`, then refresh the app.

## Main interface

1. **Forecast overview:** Interactive historical actual-vs-forecast series, hypothetical capacity threshold, hourly view, model selection.
2. **Error diagnostics:** Error distribution, residuals, fair same-period model comparison.
3. **Capacity decisions:** Capacity and reserve-margin sliders; correct/false/missed peak alerts; downloadable scenario table.
4. **Evidently monitoring:** Generate and embed/download Evidently regression report (full fixed LightGBM test set).

## Demo script (3–5 minutes)

1. Show Dec 2017 LightGBM vs actual and seasonal-naive; compare WAPE.
2. Select the busy week around peak events; discuss overshoot and missed peaks.
3. Move threshold; show recall/precision and false-alarm tradeoff.
4. Adjust reserve margin; present *illustrative* headroom calculation (not actual savings).
5. Generate/open Evidently diagnostics; explain outlier and residual review.

## Validation and limitations

- A **live user interface** is not a live data feed. The available source is historical 2017 data and model predictions are precomputed.
- Dashboard scenarios do **not** retrain the model. Threshold and margin do not cause electrical demand to change.
- The source has **no documented real capacity limit**; slider is a hypothetical scenario parameter.
- Do not treat scenario results as causal uplift, grid operations forecasts, or measured savings.
- The optional shaded band uses absolute errors from the *same displayed test period*: it is descriptive and **not** a calibrated probabilistic prediction interval.
- Evidently HTML is generated for the full test dataset and LightGBM prediction, not the selected filters.
- Next step for capstone requirement: train quantile models or calibrate prediction intervals strictly on a separate validation period, then evaluate held-out coverage and pinball loss; do not substitute the descriptive band.

## Optional standalone Evidently

```bash
python src/visualization.py
```

Open `reports/evidently_regression.html` in your browser, or access it inside the Streamlit monitoring tab.
