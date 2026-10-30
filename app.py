"""Interactive capstone dashboard; run: streamlit run app.py.

Uses existing held-out predictions. UI interactions recalculate metrics and planning
scenarios; they do NOT train a new model nor forecast future unseen observations.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import regression_metrics, peak_metrics  # noqa: E402

PREDICTIONS = ROOT / 'reports' / 'predictions.csv'
METRICS = ROOT / 'reports' / 'metrics.json'

st.set_page_config(page_title='Tetouan | Electricity Demand', page_icon='⚡', layout='wide')


@st.cache_data(show_spinner=False)
def load_outputs(csv_path: str, json_path: str, csv_mtime: float, json_mtime: float):
    df = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values('timestamp')
    metrics = json.loads(Path(json_path).read_text(encoding='utf-8'))
    required = {'timestamp', 'system_load', 'prediction_lgbm', 'prediction_seasonal_naive'}
    if not required.issubset(df.columns):
        raise ValueError(f'Predictions missing columns: {sorted(required - set(df.columns))}')
    return df, metrics


def interval_frame(df: pd.DataFrame, prediction: str, alpha: float):
    # This is an exploratory symmetric error band from TEST residuals, NOT a valid
    # calibrated prediction interval. Never label it as probabilistic coverage.
    errors = (df['system_load'] - df[prediction]).abs()
    radius = float(errors.quantile(alpha))
    return radius


def forecast_plot(df: pd.DataFrame, predicted: str, threshold: float, band: bool):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['system_load'], mode='lines',
                             name='Actual load', line=dict(color='#0f172a', width=2)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[predicted], mode='lines',
                             name='Forecast', line=dict(color='#2563eb', width=2)))
    if band:
        # Descriptive only; computed on displayed holdout sample.
        radius = interval_frame(df, predicted, 0.90)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[predicted]+radius,
                                 mode='lines', line=dict(width=0),
                                 hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[predicted]-radius,
                                 mode='lines', fill='tonexty', fillcolor='rgba(37,99,235,.10)',
                                 line=dict(width=0), name='Descriptive residual band (not calibrated)'))
    fig.add_hline(y=threshold, line_color='#ea580c', line_dash='dash',
                  annotation_text='Scenario capacity threshold')
    fig.update_layout(height=440, margin=dict(l=8, r=8, t=24, b=8),
                      xaxis_title='Timestamp (historical test set)',
                      yaxis_title='Combined system load (source measurement units)',
                      legend=dict(orientation='h', y=1.12), hovermode='x unified')
    return fig


def peak_table(df: pd.DataFrame, predicted: str, threshold: float):
    result = df[['timestamp', 'system_load', predicted]].copy()
    result['actual_breach'] = result['system_load'] >= threshold
    result['forecast_breach'] = result[predicted] >= threshold
    result['status'] = np.select(
        [result['actual_breach'] & result['forecast_breach'],
         ~result['actual_breach'] & result['forecast_breach'],
         result['actual_breach'] & ~result['forecast_breach']],
        ['Correct alert', 'False alert', 'Missed breach'], default='No alert')
    return result


if not PREDICTIONS.exists() or not METRICS.exists():
    st.error('Missing reports/predictions.csv or reports/metrics.json. Run `python src/run_pipeline.py` first.')
    st.stop()

df_all, saved_metrics = load_outputs(str(PREDICTIONS), str(METRICS),
                                      PREDICTIONS.stat().st_mtime, METRICS.stat().st_mtime)

st.title('⚡ Tetouan Electricity Demand | Capacity Planning')
st.caption('Historical 1-hour-ahead forecasts • December 2017 held-out test period • Interactive scenario analysis')

with st.sidebar:
    st.header('Demo controls')
    model_choice = st.selectbox('Forecast model', ['LightGBM', 'Seasonal naive (168h)'])
    predicted = 'prediction_lgbm' if model_choice == 'LightGBM' else 'prediction_seasonal_naive'
    lo = df_all['timestamp'].min().date()
    hi = df_all['timestamp'].max().date()
    selected_range = st.date_input('Evaluation date range', value=(lo, hi), min_value=lo, max_value=hi)
    st.divider()
    default_capacity = float(saved_metrics.get('peak_capacity_scenario', {}).get('capacity_threshold', df_all['system_load'].quantile(.8)))
    min_capacity = float(df_all['system_load'].min() * .8)
    max_capacity = float(df_all['system_load'].max() * 1.2)
    threshold = st.slider('Capacity scenario threshold', min_value=round(min_capacity, 1),
                          max_value=round(max_capacity, 1), value=round(default_capacity, 1),
                          step=100.0, help='Scenario parameter; NOT a documented physical grid capacity.')
    reserve = st.slider('Reserve margin (%)', 0, 30, 10, 1,
                        help='Illustrative headroom: max(0, forecast × (1 + reserve%) − scenario threshold).')
    show_band = st.checkbox('Show descriptive error band', value=False)
    st.caption('Predictions are loaded from a fixed historical test set. Controls never refit the model.')

if not isinstance(selected_range, tuple) or len(selected_range) != 2:
    st.info('Select a start and end date in the sidebar.')
    st.stop()
start, end = selected_range
if start > end:
    st.error('Start date must not exceed end date.')
    st.stop()
mask = (df_all['timestamp'].dt.date >= start) & (df_all['timestamp'].dt.date <= end)
df = df_all.loc[mask].copy()
if df.empty:
    st.warning('No predictions available for selected period.')
    st.stop()

metrics = regression_metrics(df['system_load'], df[predicted], df['prediction_seasonal_naive'])
peaks = peak_metrics(df['system_load'], df[predicted], threshold)
alerts = peak_table(df, predicted, threshold)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('WAPE', f"{metrics['wape']*100:.2f}%")
k2.metric('MAE', f"{metrics['mae']:,.0f}")
k3.metric('Peak recall', f"{peaks['peak_recall']*100:.1f}%" if peaks['actual_breaches'] else 'N/A')
k4.metric('Peak precision', f"{peaks['peak_precision']*100:.1f}%" if peaks['predicted_breaches'] else 'N/A')
k5.metric('Breaches anticipated', f"{peaks['breaches_correctly_anticipated']}/{peaks['actual_breaches']}")

main, diagnostics, planning, monitoring = st.tabs([
    'Forecast overview', 'Error diagnostics', 'Capacity decisions', 'Monitoring'
])
with main:
    st.plotly_chart(forecast_plot(df, predicted, threshold, show_band), width='stretch')
    st.info('The orange line is a hypothetical capacity threshold. It is not an actual documented limit of the Tetouan grid.')
    if show_band:
        st.warning('The shaded band is a descriptive TEST-sample residual range, NOT a calibrated forecast interval. Do not use it to claim 90% future coverage.')
    st.subheader('Selected-period breakdown')
    hour = df.assign(hour=df['timestamp'].dt.hour).groupby('hour', as_index=False).agg(
        actual_load=('system_load','mean'), forecast_load=(predicted,'mean'))
    st.line_chart(hour.set_index('hour'), y=['actual_load', 'forecast_load'])

with diagnostics:
    residual = df['system_load'] - df[predicted]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=residual, nbinsx=35, marker_color='#2563eb'))
        fig.update_layout(title='Forecast error distribution', xaxis_title='Actual − forecast', yaxis_title='Hours', height=320)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = go.Figure(go.Scatter(x=df[predicted], y=residual, mode='markers',
                                   marker=dict(color='#0d9488', opacity=.50, size=5)))
        fig.add_hline(y=0, line_dash='dash')
        fig.update_layout(title='Residual vs forecast', xaxis_title='Predicted load', yaxis_title='Residual', height=320)
        st.plotly_chart(fig, width='stretch')
    comparison = pd.DataFrame([
        {'model':'LightGBM', **regression_metrics(df['system_load'], df['prediction_lgbm'])},
        {'model':'Seasonal naive', **regression_metrics(df['system_load'], df['prediction_seasonal_naive'])},
    ])
    comparison['WAPE (%)'] = comparison['wape'] * 100
    st.dataframe(comparison[['model','mae','rmse','WAPE (%)']], width='stretch', hide_index=True)
    if len(df) > 1:
        st.caption('Metrics are computed only over the selected subset. Narrow slices can produce unstable conclusions.')

with planning:
    st.subheader('Capacity threshold decision simulator')
    st.caption('Alerts and confusion counts are retrospective, based on historical labels available after the forecast hour.')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Correct alerts', int((alerts['status']=='Correct alert').sum()))
    c2.metric('False alerts', int((alerts['status']=='False alert').sum()))
    c3.metric('Missed breaches', int((alerts['status']=='Missed breach').sum()))
    c4.metric('Forecast alerts', int(alerts['forecast_breach'].sum()))
    alerts['suggested_reserve'] = np.maximum(0, df[predicted].to_numpy() * (1 + reserve/100) - threshold)
    st.metric('Maximum suggested extra reserve (scenario units)', f"{alerts['suggested_reserve'].max():,.0f}")
    with st.expander('View hour-by-hour alerts and reserve suggestions', expanded=True):
        display = alerts.loc[(alerts['forecast_breach'] | alerts['actual_breach'])].copy()
        if display.empty:
            st.info('No threshold breaches or alerts in this selection. Lower the threshold to demonstrate the trade-off.')
        else:
            st.dataframe(display, width='stretch', hide_index=True)
        st.download_button('Download selected scenario CSV', alerts.to_csv(index=False).encode('utf-8'),
                           file_name='capacity_scenario.csv', mime='text/csv')

    st.subheader('Sensitivity analysis')
    st.caption('Threshold sweep is anchored on the training-set 80th-percentile threshold; reserve sweep varies the illustrative margin. These are test-set scenario comparisons, not expected savings or operational forecasts.')
    reference_threshold = float(saved_metrics.get('peak_capacity_scenario', {}).get('capacity_threshold', threshold))
    threshold_grid = np.linspace(reference_threshold * .85, reference_threshold * 1.15, 31)
    sensitivity_rows = []
    for test_threshold in threshold_grid:
        result = peak_metrics(df['system_load'], df[predicted], float(test_threshold))
        tp = result['breaches_correctly_anticipated']
        sensitivity_rows.append({
            'threshold': test_threshold,
            'precision (%)': (result['peak_precision'] or 0) * 100,
            'recall (%)': (result['peak_recall'] or 0) * 100,
            'actual breaches': result['actual_breaches'],
            'false alerts': result['predicted_breaches'] - tp,
            'missed breaches': result['actual_breaches'] - tp,
        })
    threshold_sensitivity = pd.DataFrame(sensitivity_rows)
    sens_col, reserve_col = st.columns(2)
    with sens_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=threshold_sensitivity['threshold'], y=threshold_sensitivity['precision (%)'], name='Precision', mode='lines'))
        fig.add_trace(go.Scatter(x=threshold_sensitivity['threshold'], y=threshold_sensitivity['recall (%)'], name='Recall', mode='lines'))
        fig.add_vline(x=threshold, line_dash='dash', line_color='#ea580c', annotation_text='Current threshold')
        fig.update_layout(title='Peak alert trade-off by threshold', xaxis_title='Scenario threshold (source units)', yaxis_title='Percent', height=350)
        st.plotly_chart(fig, width='stretch')
    with reserve_col:
        reserve_grid = np.arange(0, 31, 5)
        reserve_sensitivity = pd.DataFrame([
            {
                'reserve margin (%)': margin,
                'hours with extra reserve': int((df[predicted] * (1 + margin / 100) > threshold).sum()),
                'maximum extra reserve': float(np.maximum(0, df[predicted] * (1 + margin / 100) - threshold).max()),
            }
            for margin in reserve_grid
        ])
        fig = go.Figure(go.Scatter(x=reserve_sensitivity['reserve margin (%)'],
                                   y=reserve_sensitivity['maximum extra reserve'],
                                   mode='lines+markers', name='Maximum extra reserve'))
        fig.update_layout(title='Reserve requirement by margin', xaxis_title='Reserve margin (%)',
                          yaxis_title='Maximum extra reserve (source units)', height=350)
        st.plotly_chart(fig, width='stretch')
    with st.expander('Sensitivity values'):
        st.dataframe(threshold_sensitivity, width='stretch', hide_index=True)
        st.dataframe(reserve_sensitivity, width='stretch', hide_index=True)
    st.warning('This is a scenario comparison, not a causal claim or operational recommendation. No true grid capacity, intervention costs or measured savings are provided by this dataset.')

with monitoring:
    st.subheader('Historical forecast monitoring')
    st.caption('This view surfaces the same kinds of regression quality and data-summary details as the Evidently report, calculated in the selected period. It is retrospective monitoring, not live telemetry or production drift detection.')
    monitor = df[['timestamp', 'system_load', predicted]].copy()
    monitor['error'] = monitor[predicted] - monitor['system_load']
    monitor['absolute_percentage_error'] = (
        monitor['error'].abs() / monitor['system_load'].abs().replace(0, np.nan) * 100
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('R²', f"{r2_score(monitor['system_load'], monitor[predicted]):.3f}" if len(monitor) > 1 else 'N/A')
    m2.metric('Mean error (forecast − actual)', f"{monitor['error'].mean():,.1f}")
    m3.metric('MAPE', f"{monitor['absolute_percentage_error'].mean():.2f}%")
    m4.metric('Maximum absolute error', f"{monitor['error'].abs().max():,.1f}")

    daily_rows = []
    for day, part in monitor.groupby(monitor['timestamp'].dt.floor('D'), sort=True):
        daily_rows.append({
            'date': day,
            'mean_error': part['error'].mean(),
            'mae': part['error'].abs().mean(),
            'mape (%)': part['absolute_percentage_error'].mean(),
        })
    daily = pd.DataFrame(daily_rows)
    daily_metric = st.selectbox('Daily error metric', ['mean_error', 'mae', 'mape (%)'], key='daily_monitor_metric')
    daily_fig = go.Figure(go.Scatter(x=daily['date'], y=daily[daily_metric], mode='lines+markers'))
    if daily_metric == 'mean_error':
        daily_fig.add_hline(y=0, line_dash='dash', line_color='#64748b')
    daily_fig.update_layout(height=330, xaxis_title='Date', yaxis_title=daily_metric,
                            margin=dict(l=8, r=8, t=24, b=8))
    st.plotly_chart(daily_fig, width='stretch')

    dist_col, ape_col = st.columns(2)
    with dist_col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=monitor['system_load'], name='Actual', opacity=.6))
        fig.add_trace(go.Histogram(x=monitor[predicted], name='Forecast', opacity=.6))
        fig.update_layout(title='Actual and forecast distributions', barmode='overlay',
                          xaxis_title='Load (source measurement units)', yaxis_title='Hours', height=330)
        st.plotly_chart(fig, width='stretch')
    with ape_col:
        fig = go.Figure(go.Histogram(x=monitor['absolute_percentage_error'], nbinsx=30,
                                     marker_color='#7c3aed'))
        fig.update_layout(title='Absolute percentage error distribution',
                          xaxis_title='Absolute percentage error (%)', yaxis_title='Hours', height=330)
        st.plotly_chart(fig, width='stretch')

    monitor['week'] = monitor['timestamp'].dt.to_period('W').dt.start_time
    weekly_rows = []
    for week, part in monitor.groupby('week', sort=True):
        weekly_rows.append({
            'week': week,
            **regression_metrics(part['system_load'], part[predicted]),
        })
    weekly = pd.DataFrame(weekly_rows)
    if weekly.empty:
        st.info('Not enough data in the selected range to calculate monitoring metrics.')
    else:
        weekly['WAPE (%)'] = weekly['wape'] * 100
        weekly['MAE'] = weekly['mae']
        metric_choice = st.selectbox('Metric over time', ['WAPE (%)', 'MAE'], key='monitor_metric')
        fig = go.Figure(go.Scatter(x=weekly['week'], y=weekly[metric_choice], mode='lines+markers'))
        fig.update_layout(height=360, xaxis_title='Week', yaxis_title=metric_choice,
                          margin=dict(l=8, r=8, t=24, b=8))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(weekly[['week', 'mae', 'rmse', 'WAPE (%)']], width='stretch', hide_index=True)
        st.download_button('Download weekly metrics CSV', weekly.to_csv(index=False).encode('utf-8'),
                           file_name='weekly_monitoring_metrics.csv', mime='text/csv')

    with st.expander('Dataset summary and field distributions'):
        summary_cols = ['system_load', predicted]
        if 'capacity_threshold' in df.columns:
            summary_cols.append('capacity_threshold')
        numeric_summary = df[summary_cols].describe().T
        numeric_summary['missing'] = df[summary_cols].isna().sum()
        st.markdown('**Numeric fields**')
        st.dataframe(numeric_summary, width='stretch')
        field_summary = pd.DataFrame({
            'dtype': df.dtypes.astype(str),
            'non-missing': df.notna().sum(),
            'missing': df.isna().sum(),
            'unique': df.nunique(dropna=True),
        })
        st.markdown('**All fields**')
        st.dataframe(field_summary, width='stretch')
        st.caption(f"Rows: {len(df):,} · Columns: {len(df.columns)} · Missing values: {int(df.isna().sum().sum())} · Duplicate rows: {int(df.duplicated().sum())}")
        peak_counts = pd.DataFrame({
            'Actual peak': alerts['actual_breach'].value_counts().reindex([False, True], fill_value=0),
            'Forecast peak': alerts['forecast_breach'].value_counts().reindex([False, True], fill_value=0),
        }, index=[False, True]).rename(index={False: 'Below threshold', True: 'At/above threshold'})
        st.markdown('**Peak label distribution at the selected scenario threshold**')
        st.bar_chart(peak_counts)

st.divider()
st.caption('Teaching baseline • 1-hour-ahead hourly forecasting • source load units unverified • 2017 data ≠ live grid telemetry')
