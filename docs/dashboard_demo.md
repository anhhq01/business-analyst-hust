# Streamlit dashboard demonstration

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the browser at `http://localhost:8501`. Precomputed December 2017 predictions are packaged, so you **do not** have to rerun model training to demonstrate the dashboard. For fresh historical retraining: `python src/run_pipeline.py`, then refresh the app.

## Evidently UI

To store the fixed test-period evaluation in Evidently's local UI workspace and open the Evidently dashboard, run these commands in a separate terminal:

```powershell
python src/visualization.py
```

Copy and run the PowerShell command printed by the script in a second terminal, then open `http://localhost:8000`. It sets `DO_NOT_TRACK=1` to avoid a telemetry write to restricted AppData. Each run of `src/visualization.py` adds a snapshot to the `Tetouan hourly load evaluation` project. The workspace is stored under `%TEMP%\Tetouan_Evidently_UI`; keep that folder to preserve snapshots. These are historical batch evaluations, not live monitoring runs.

The Evidently dashboard shows regression quality, data quality, actual/forecast load summaries, and peak-label counts. Open an individual stored report in the project to see the detailed regression and data-summary visualizations, including error distributions.

## Main interface

1. **Forecast overview:** Interactive historical actual-vs-forecast series, hypothetical capacity threshold, hourly view, model selection.
2. **Error diagnostics:** Error distribution, residuals, fair same-period model comparison.
3. **Capacity decisions:** Capacity and reserve-margin sliders; correct/false/missed peak alerts; downloadable scenario table; threshold and reserve-margin sensitivity sweeps.
4. **Monitoring:** Review R², MAPE, maximum error, daily/weekly error trends, dataset summaries, and download weekly metrics.

## Demo script (3–5 minutes)

1. Show Dec 2017 LightGBM vs actual and seasonal-naive; compare WAPE.
2. Select the busy week around peak events; discuss overshoot and missed peaks.
3. Move threshold; show recall/precision and false-alarm tradeoff.
4. Adjust reserve margin; present *illustrative* headroom calculation (not actual savings).
5. Compare weekly errors and discuss that the data is historical rather than live telemetry.

Read [`executive_brief.md`](executive_brief.md) for the recommendation and business limitations. Sensitivity outputs are historical scenario comparisons and do not represent expected savings.

## Validation and limitations

- A **live user interface** is not a live data feed. The available source is historical 2017 data and model predictions are precomputed.
- Dashboard scenarios do **not** retrain the model. Threshold and margin do not cause electrical demand to change.
- The source has **no documented real capacity limit**; slider is a hypothetical scenario parameter.
- Do not treat scenario results as causal uplift, grid operations forecasts, or measured savings.
- The optional shaded band uses absolute errors from the *same displayed test period*: it is descriptive and **not** a calibrated probabilistic prediction interval.
- Next step for capstone requirement: train quantile models or calibrate prediction intervals strictly on a separate validation period, then evaluate held-out coverage and pinball loss; do not substitute the descriptive band.
