"""Interactive capstone dashboard; run: streamlit run app.py.

Uses existing held-out predictions, calibrated quantile intervals (P10/P50/P90),
rolling-origin backtest results, multi-horizon & zone comparisons, and TreeSHAP
attribution. UI interactions recalculate metrics and planning scenarios without
retraining a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import (  # noqa: E402
    regression_metrics,
    peak_metrics,
    probabilistic_metrics,
)

PREDICTIONS = ROOT / 'reports' / 'predictions.csv'
METRICS = ROOT / 'reports' / 'metrics.json'
BACKTEST_METRICS = ROOT / 'reports' / 'backtest_metrics.json'
BACKTEST_PREDICTIONS = ROOT / 'reports' / 'backtest_predictions.csv'
SENSITIVITY_CSV = ROOT / 'reports' / 'threshold_sensitivity.csv'
HORIZON_METRICS = ROOT / 'reports' / 'multi_horizon_metrics.json'
ZONE_METRICS = ROOT / 'reports' / 'zone_comparison_metrics.json'
SHAP_SUMMARY_CSV = ROOT / 'reports' / 'shap_importance.csv'
SHAP_LOCAL_JSON = ROOT / 'reports' / 'shap_peak_explanations.json'
DQ_SCORECARD_JSON = ROOT / 'data' / 'processed' / 'dq_scorecard.json'
EVIDENTLY_HTML = ROOT / 'reports' / 'evidently_regression.html'

st.set_page_config(page_title='Tetouan | Electricity Demand & Peak Planning', page_icon='⚡', layout='wide')


@st.cache_data(show_spinner=False)
def load_outputs(csv_path: str, json_path: str, csv_mtime: float, json_mtime: float):
    df = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values('timestamp')
    metrics = json.loads(Path(json_path).read_text(encoding='utf-8'))
    required = {'timestamp', 'system_load', 'prediction_lgbm', 'prediction_seasonal_naive'}
    if not required.issubset(df.columns):
        raise ValueError(f'Predictions missing columns: {sorted(required - set(df.columns))}')
    return df, metrics


def load_optional_json(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return None


def forecast_plot(
    df: pd.DataFrame,
    predicted: str,
    threshold: float,
    interval_mode: str,
):
    fig = go.Figure()
    if interval_mode == 'Calibrated P10–P90 (80% CQR)' and {'prediction_lgbm_p10', 'prediction_lgbm_p90'}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['prediction_lgbm_p90'],
            mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['prediction_lgbm_p10'],
            mode='lines', fill='tonexty', fillcolor='rgba(37,99,235,0.16)',
            line=dict(width=0), name='Calibrated P10–P90 Forecast Interval (80%)',
        ))
    elif interval_mode == 'Raw Quantile P10–P90' and {'prediction_lgbm_p10_raw', 'prediction_lgbm_p90_raw'}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['prediction_lgbm_p90_raw'],
            mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['prediction_lgbm_p10_raw'],
            mode='lines', fill='tonexty', fillcolor='rgba(13,148,136,0.16)',
            line=dict(width=0), name='Raw LightGBM P10–P90 Interval',
        ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['system_load'], mode='lines',
        name='Actual system load', line=dict(color='#0f172a', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[predicted], mode='lines',
        name='Selected forecast', line=dict(color='#2563eb', width=2),
    ))
    fig.add_hline(
        y=threshold, line_color='#ea580c', line_dash='dash',
        annotation_text='Scenario capacity threshold',
    )
    fig.update_layout(
        height=440, margin=dict(l=8, r=8, t=24, b=8),
        xaxis_title='Timestamp (historical test set)',
        yaxis_title='Combined system load (source measurement units)',
        legend=dict(orientation='h', y=1.12), hovermode='x unified',
    )
    return fig


def peak_table(df: pd.DataFrame, predicted: str, threshold: float):
    cols = ['timestamp', 'system_load', predicted]
    for extra in ['prediction_lgbm_p10', 'prediction_lgbm_p90']:
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    result = df[cols].copy()
    result['actual_breach'] = result['system_load'] >= threshold
    result['forecast_breach'] = result[predicted] >= threshold
    result['status'] = np.select(
        [
            result['actual_breach'] & result['forecast_breach'],
            ~result['actual_breach'] & result['forecast_breach'],
            result['actual_breach'] & ~result['forecast_breach'],
        ],
        ['Correct alert', 'False alert', 'Missed breach'],
        default='No alert',
    )
    return result


if not PREDICTIONS.exists() or not METRICS.exists():
    st.error('Missing reports/predictions.csv or reports/metrics.json. Run `python src/run_pipeline.py` first.')
    st.stop()

df_all, saved_metrics = load_outputs(
    str(PREDICTIONS), str(METRICS),
    PREDICTIONS.stat().st_mtime, METRICS.stat().st_mtime,
)

st.title('⚡ Tetouan Electricity Demand | Probabilistic Forecasting & Peak Capacity Planning')
st.caption('Capstone Topic #8 • 1-Hour-Ahead Point & Quantile Forecasts (P10/P50/P90) • Rolling Backtesting • Risk-Aware Reserve Simulation')

model_map = {
    'LightGBM Point (L1)': 'prediction_lgbm',
    'Seasonal Naive (168h)': 'prediction_seasonal_naive',
}
if 'prediction_lgbm_p50' in df_all.columns:
    model_map['LightGBM Median (P50)'] = 'prediction_lgbm_p50'
if 'prediction_lgbm_p90' in df_all.columns:
    model_map['LightGBM Risk-Aware Upper (P90)'] = 'prediction_lgbm_p90'
if 'prediction_ridge' in df_all.columns:
    model_map['Ridge Regression (Standardized)'] = 'prediction_ridge'
if 'prediction_seasonal_naive_24h' in df_all.columns:
    model_map['Seasonal Naive (24h)'] = 'prediction_seasonal_naive_24h'

with st.sidebar:
    st.header('Decision & Scenario Controls')
    model_choice = st.selectbox('Primary forecast series', list(model_map.keys()))
    predicted = model_map[model_choice]

    lo = df_all['timestamp'].min().date()
    hi = df_all['timestamp'].max().date()
    selected_range = st.date_input('Evaluation date range', value=(lo, hi), min_value=lo, max_value=hi)

    st.divider()
    train_pcts = saved_metrics.get('threshold_percentiles_train', {})
    preset_choice = st.selectbox(
        'Capacity threshold preset (from training distribution)',
        ['P80 (Baseline)', 'P90', 'P95', 'P97.5', 'Custom slider'],
    )
    default_capacity = float(saved_metrics.get('peak_capacity_scenario', {}).get('capacity_threshold', df_all['system_load'].quantile(0.8)))
    if preset_choice == 'P80 (Baseline)':
        preset_val = float(train_pcts.get('P80', default_capacity))
    elif preset_choice == 'P90':
        preset_val = float(train_pcts.get('P90', default_capacity * 1.05))
    elif preset_choice == 'P95':
        preset_val = float(train_pcts.get('P95', default_capacity * 1.08))
    elif preset_choice == 'P97.5':
        preset_val = float(train_pcts.get('P97.5', default_capacity * 1.10))
    else:
        preset_val = default_capacity

    min_capacity = float(df_all['system_load'].min() * 0.8)
    max_capacity = float(df_all['system_load'].max() * 1.2)
    threshold = st.slider(
        'Capacity scenario threshold',
        min_value=round(min_capacity, 1),
        max_value=round(max_capacity, 1),
        value=round(preset_val, 1),
        step=100.0,
        help='Scenario parameter derived from training distribution; NOT a physical limit of Tetouan grid.',
    )
    reserve = st.slider(
        'Reserve margin (%)', 0, 30, 10, 1,
        help='Extra reserve rule: max(0, forecast × (1 + reserve%) − scenario threshold).',
    )
    unmet_penalty_ratio = st.slider(
        'Asymmetric penalty ratio (Unmet Peak : Reserve Cost)', 2, 25, 10, 1,
        help='Scenario cost weight for unserved peak load above dispatched reserve relative to unit reserve cost.',
    )
    interval_mode = st.selectbox(
        'Uncertainty band on chart',
        ['Calibrated P10–P90 (80% CQR)', 'Raw Quantile P10–P90', 'Hidden'],
    )
    st.caption('All thresholds and intervals are calibrated on pre-test data only (zero test leakage).')

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
peaks = peak_metrics(df['system_load'], df[predicted], threshold, timestamps=df['timestamp'])
alerts = peak_table(df, predicted, threshold)

has_quantiles = {'prediction_lgbm_p10', 'prediction_lgbm_p50', 'prediction_lgbm_p90'}.issubset(df.columns)
prob_stats = (
    probabilistic_metrics(
        df['system_load'], df['prediction_lgbm_p10'], df['prediction_lgbm_p50'], df['prediction_lgbm_p90']
    )
    if has_quantiles else None
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric('WAPE', f"{metrics['wape']*100:.2f}%", f"MASE {metrics['mase_vs_seasonal_naive']:.3f}")
k2.metric('Peak Recall', f"{peaks['peak_recall']*100:.1f}%" if peaks['actual_breaches'] else 'N/A',
          f"{peaks['breaches_correctly_anticipated']}/{peaks['actual_breaches']} breaches")
k3.metric('Peak Precision', f"{peaks['peak_precision']*100:.1f}%" if peaks['predicted_breaches'] else 'N/A',
          f"{peaks['false_alarms']} false alerts")
k4.metric('Peak Timing Error', f"{peaks.get('mean_abs_peak_timing_error_hours', 0.0):.2f} h",
          f"{peaks.get('within_1h_peak_timing_rate', 0.0)*100:.0f}% within ±1h")
if prob_stats:
    k5.metric('P10–P90 Coverage', f"{prob_stats['empirical_coverage']*100:.1f}%", 'Target: 80.0%')
    k6.metric('Mean Pinball Loss', f"{prob_stats['mean_pinball_loss']:,.1f}",
              f"Width: {prob_stats['mean_interval_width']:,.0f}")
else:
    k5.metric('MAE', f"{metrics['mae']:,.0f}")
    k6.metric('RMSE', f"{metrics['rmse']:,.0f}")

tab_overview, tab_planning, tab_backtest, tab_shap, tab_monitoring = st.tabs([
    '1. Forecast & Intervals',
    '2. Capacity & Sensitivity',
    '3. Backtesting, Horizons & Zones',
    '4. Explainability (SHAP) & Diagnostics',
    '5. Data Quality & Evidently',
])

with tab_overview:
    st.plotly_chart(forecast_plot(df, predicted, threshold, interval_mode), width='stretch')
    st.info(
        'The shaded band shows the **out-of-sample P10–P90 probabilistic forecast interval** '
        '(trained with Quantile LightGBM and calibrated on the November 2017 validation split). '
        'The dashed orange line is a scenario capacity threshold.'
    )
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader('Hourly Diurnal Load & Peak Window Profile')
        hour_cols = {'actual_load': ('system_load', 'mean'), 'forecast_load': (predicted, 'mean')}
        if 'prediction_lgbm_p90' in df.columns:
            hour_cols['p90_upper_bound'] = ('prediction_lgbm_p90', 'mean')
        hour = df.assign(hour=df['timestamp'].dt.hour).groupby('hour', as_index=False).agg(**hour_cols)
        st.line_chart(hour.set_index('hour'))
    with c_right:
        st.subheader('Model Benchmark Comparison (Selected Window)')
        comp_rows = []
        for label, col in model_map.items():
            rm = regression_metrics(df['system_load'], df[col], df['prediction_seasonal_naive'])
            pm = peak_metrics(df['system_load'], df[col], threshold, timestamps=df['timestamp'])
            comp_rows.append({
                'Model / Policy': label,
                'MAE': round(rm['mae'], 1),
                'RMSE': round(rm['rmse'], 1),
                'WAPE (%)': round(rm['wape'] * 100, 2),
                'MASE': round(rm['mase_vs_seasonal_naive'], 3),
                'Peak Recall (%)': round(pm['peak_recall'] * 100, 1) if pm['peak_recall'] is not None else None,
                'Peak Timing Err (h)': round(pm.get('mean_abs_peak_timing_error_hours', np.nan), 2),
            })
        st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)

with tab_planning:
    st.subheader('Risk-Aware Capacity Threshold & Reserve-Margin Simulator')
    st.caption(
        'Compares dispatching standby reserve using Point Forecasts (`P50`) vs. Risk-Aware Upper Quantiles (`P90`) '
        'under explicit reserve margins and asymmetric unmet-demand penalties.'
    )
    alerts['suggested_reserve'] = np.maximum(0.0, df[predicted].to_numpy() * (1 + reserve / 100.0) - threshold)
    alerts['effective_capacity'] = threshold + alerts['suggested_reserve']
    alerts['unmet_shortfall'] = np.maximum(0.0, df['system_load'].to_numpy() - alerts['effective_capacity'])
    scenario_reserve_cost = float(alerts['suggested_reserve'].sum() * 1.0)
    scenario_shortfall_cost = float(alerts['unmet_shortfall'].sum() * unmet_penalty_ratio)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Correct Alerts (TP)', int((alerts['status'] == 'Correct alert').sum()))
    c2.metric('False Alerts (FP)', int((alerts['status'] == 'False alert').sum()))
    c3.metric('Missed Breaches (FN)', int((alerts['status'] == 'Missed breach').sum()))
    c4.metric('Unmet Peak Units', f"{alerts['unmet_shortfall'].sum():,.0f}")
    c5.metric(f'Total Scenario Cost ({unmet_penalty_ratio}:1)', f"{scenario_reserve_cost + scenario_shortfall_cost:,.0f}")

    if SENSITIVITY_CSV.exists():
        st.markdown('#### Multi-Threshold & Reserve Policy Sensitivity Table (P80 / P90 / P95 / P97.5)')
        sens_df = pd.read_csv(SENSITIVITY_CSV)
        filter_pct = st.multiselect(
            'Filter threshold percentiles',
            options=sens_df['threshold_percentile'].unique().tolist(),
            default=['P80', 'P90', 'P95'],
        )
        st.dataframe(
            sens_df[sens_df['threshold_percentile'].isin(filter_pct)],
            width='stretch', hide_index=True,
        )

    with st.expander('View hour-by-hour alerts and reserve dispatch schedule', expanded=False):
        display = alerts.loc[(alerts['forecast_breach'] | alerts['actual_breach'])].copy()
        if display.empty:
            st.info('No threshold breaches or alerts in this selection. Lower the threshold to inspect trade-offs.')
        else:
            st.dataframe(display, width='stretch', hide_index=True)
        st.download_button(
            'Download selected scenario CSV',
            alerts.to_csv(index=False).encode('utf-8'),
            file_name='capacity_scenario.csv',
            mime='text/csv',
        )
    st.warning(
        'Scope Guardrail: This is a transparent scenario simulation, not a causal claim or physical grid guarantee. '
        'True plant commitment costs and physical transformer limits are not observed in the UCI dataset.'
    )

with tab_backtest:
    st.subheader('Rolling-Origin Backtesting, Multi-Horizon & Zone Subgroup Analysis')
    bt_data = load_optional_json(BACKTEST_METRICS)
    hz_data = load_optional_json(HORIZON_METRICS)
    zn_data = load_optional_json(ZONE_METRICS)

    if bt_data:
        st.markdown('#### 1. 5-Fold Expanding-Window Rolling Backtest (Aug–Dec 2017)')
        bt_folds = pd.DataFrame(bt_data['folds'])
        st.dataframe(
            bt_folds[[
                'fold', 'test_start', 'test_end', 'lightgbm_wape', 'ridge_wape',
                'seasonal_naive_wape', 'lightgbm_mase', 'empirical_coverage_80',
                'peak_recall_p80', 'mean_abs_peak_timing_error_hours',
            ]],
            width='stretch', hide_index=True,
        )
        agg = bt_data['aggregate']
        st.caption(
            f"5-Fold Mean LightGBM WAPE: **{agg['lightgbm_wape_mean']*100:.2f}% ± {agg['lightgbm_wape_std']*100:.2f}%** | "
            f"Mean MASE: **{agg['lightgbm_mase_mean']:.3f}** | "
            f"Mean 80% Interval Coverage: **{agg['empirical_coverage_80_mean']*100:.1f}%** | "
            f"Mean Peak Timing Error: **{agg['mean_abs_peak_timing_error_hours']:.2f} h**"
        )

    col_hz, col_zn = st.columns(2)
    with col_hz:
        if hz_data:
            st.markdown('#### 2. Multi-Horizon Forecasting (`1h` vs `6h` vs `24h` Ahead)')
            hz_rows = []
            for key, val in hz_data.items():
                hz_rows.append({
                    'Horizon': f"{val['horizon_hours']}h ahead",
                    'Min Lag Used': f"t - {val['min_feature_lag_hours']}h",
                    'LightGBM WAPE (%)': round(val['lightgbm']['wape'] * 100, 2),
                    'Ridge WAPE (%)': round(val['ridge']['wape'] * 100, 2),
                    'MASE vs 168h': round(val['lightgbm']['mase_vs_seasonal_naive'], 3),
                    '80% Coverage (%)': round(val['probabilistic']['empirical_coverage'] * 100, 1),
                    'Peak Recall (%)': (
                        round(val['peak_metrics']['peak_recall'] * 100, 1)
                        if val['peak_metrics']['peak_recall'] is not None else None
                    ),
                    'Peak Timing Err (h)': round(val['peak_metrics']['mean_abs_peak_timing_error_hours'], 2),
                })
            st.dataframe(pd.DataFrame(hz_rows), width='stretch', hide_index=True)

    with col_zn:
        if zn_data:
            st.markdown('#### 3. Zone-Specific vs. Combined Model & Subgroup Fairness')
            z_rows = [{
                'Scope / Subgroup': 'Combined System (Direct Model)',
                'Load Share (%)': 100.0,
                'LightGBM WAPE (%)': round(zn_data['direct_combined_model']['wape'] * 100, 2),
                'MASE': round(zn_data['direct_combined_model']['mase_vs_seasonal_naive'], 3),
                'Peak Recall P80 (%)': (
                    round(zn_data['direct_combined_model']['peak_metrics_p80']['peak_recall'] * 100, 1)
                    if zn_data['direct_combined_model']['peak_metrics_p80']['peak_recall'] is not None else None
                ),
                'Peak Timing Err (h)': round(zn_data['direct_combined_model']['peak_metrics_p80']['mean_abs_peak_timing_error_hours'], 2),
            }, {
                'Scope / Subgroup': 'Combined System (Bottom-Up Sum of Zones)',
                'Load Share (%)': 100.0,
                'LightGBM WAPE (%)': round(zn_data['bottom_up_zone_sum_model']['wape'] * 100, 2),
                'MASE': round(zn_data['bottom_up_zone_sum_model']['mase_vs_seasonal_naive'], 3),
                'Peak Recall P80 (%)': (
                    round(zn_data['bottom_up_zone_sum_model']['peak_metrics_p80']['peak_recall'] * 100, 1)
                    if zn_data['bottom_up_zone_sum_model']['peak_metrics_p80']['peak_recall'] is not None else None
                ),
                'Peak Timing Err (h)': round(zn_data['bottom_up_zone_sum_model']['peak_metrics_p80']['mean_abs_peak_timing_error_hours'], 2),
            }]
            for z_name, z_val in zn_data['zones'].items():
                z_rec = z_val['peak_metrics_p80']['peak_recall']
                z_rows.append({
                    'Scope / Subgroup': z_name,
                    'Load Share (%)': round(z_val['share_of_system_load_pct'], 1),
                    'LightGBM WAPE (%)': round(z_val['lightgbm']['wape'] * 100, 2),
                    'MASE': round(z_val['lightgbm']['mase_vs_seasonal_naive'], 3),
                    'Peak Recall P80 (%)': f"{z_rec * 100:.1f}%" if z_rec is not None else 'N/A (0 Dec breaches)',
                    'Peak Timing Err (h)': round(z_val['peak_metrics_p80']['mean_abs_peak_timing_error_hours'], 2),
                })
            for r in z_rows[:2]:
                if isinstance(r['Peak Recall P80 (%)'], (int, float)):
                    r['Peak Recall P80 (%)'] = f"{r['Peak Recall P80 (%)']:.1f}%"
            st.dataframe(pd.DataFrame(z_rows), width='stretch', hide_index=True)
            st.caption('Note: Zone 3 peaks strongly in July–August and has 0 breaches of its full-year P80 threshold during December.')

with tab_shap:
    st.subheader('Model Explainability (TreeSHAP & Permutation Importance) & Residual Diagnostics')
    if SHAP_SUMMARY_CSV.exists():
        shap_df = pd.read_csv(SHAP_SUMMARY_CSV).head(15)
        s1, s2 = st.columns(2)
        with s1:
            fig_shap = go.Figure(go.Bar(
                x=shap_df['mean_abs_shap'][::-1],
                y=shap_df['feature'][::-1],
                orientation='h',
                marker_color='#2563eb',
                name='Mean |TreeSHAP|',
            ))
            fig_shap.update_layout(
                title='Top 15 Features by Mean |TreeSHAP| Contribution (Test Set)',
                xaxis_title='Mean |SHAP value| (load units)',
                height=420, margin=dict(l=8, r=8, t=36, b=8),
            )
            st.plotly_chart(fig_shap, width='stretch')
        with s2:
            shap_local = load_optional_json(SHAP_LOCAL_JSON)
            if shap_local:
                peak_options = [f"{item['timestamp']} (Actual: {item['actual_load']:,.0f})" for item in shap_local]
                chosen_idx = st.selectbox('Inspect local TreeSHAP explanation for top peak hour', range(len(peak_options)), format_func=lambda i: peak_options[i])
                local_item = shap_local[chosen_idx]
                contrib_df = pd.DataFrame(local_item['top_contributions'])
                fig_loc = go.Figure(go.Bar(
                    x=contrib_df['shap_contribution'][::-1],
                    y=contrib_df['feature'][::-1],
                    orientation='h',
                    marker_color=['#dc2626' if v > 0 else '#0d9488' for v in contrib_df['shap_contribution'][::-1]],
                ))
                fig_loc.update_layout(
                    title=f"Local SHAP Drivers at {local_item['timestamp']}",
                    xaxis_title='SHAP contribution above/below base load',
                    height=360, margin=dict(l=8, r=8, t=36, b=8),
                )
                st.plotly_chart(fig_loc, width='stretch')
        with st.expander('View Global TreeSHAP vs. Permutation Importance (ΔMAE) & Split Gain Table', expanded=False):
            st.dataframe(shap_df, width='stretch', hide_index=True)

    residual = df['system_load'] - df[predicted]
    d1, d2 = st.columns(2)
    with d1:
        fig_hist = go.Figure(go.Histogram(x=residual, nbinsx=35, marker_color='#2563eb'))
        fig_hist.update_layout(title='Forecast Error Distribution (Actual − Forecast)', xaxis_title='Residual', yaxis_title='Hours', height=320)
        st.plotly_chart(fig_hist, width='stretch')
    with d2:
        fig_res = go.Figure(go.Scatter(
            x=df[predicted], y=residual, mode='markers',
            marker=dict(color='#0d9488', opacity=0.5, size=5),
        ))
        fig_res.add_hline(y=0, line_dash='dash')
        fig_res.update_layout(title='Residuals vs. Fitted (Heteroscedasticity Check)', xaxis_title='Predicted Load', yaxis_title='Residual', height=320)
        st.plotly_chart(fig_res, width='stretch')

with tab_monitoring:
    dq = load_optional_json(DQ_SCORECARD_JSON)
    if dq:
        st.subheader('6-Dimension Data Quality Scorecard (BA3 Framework)')
        dq_rows = [
            {
                'Dimension': k.replace('_', ' ').title(),
                'Metric': v['metric'],
                'Observed Value': str(v['value']),
                'SLA Threshold': v['sla_threshold'],
                'Status': v['status'],
            }
            for k, v in dq['dimensions'].items()
        ]
        st.dataframe(pd.DataFrame(dq_rows), width='stretch', hide_index=True)

    st.subheader('Evidently Regression Monitoring Report')
    if EVIDENTLY_HTML.exists():
        st.success('Saved Evidently HTML report found (`reports/evidently_regression.html`).')
        st.download_button(
            'Download Evidently HTML',
            EVIDENTLY_HTML.read_bytes(),
            file_name='evidently_regression.html',
            mime='text/html',
        )
        if st.checkbox('Embed saved Evidently report', value=False):
            if hasattr(st, 'iframe'):
                st.iframe(EVIDENTLY_HTML, height=850)
            else:
                from streamlit.components.v1 import html
                html(EVIDENTLY_HTML.read_text(encoding='utf-8'), height=850, scrolling=True)
    else:
        st.info('No Evidently report generated yet. Click below to build one.')
    if st.button('Generate / refresh Evidently report'):
        try:
            from visualization import build_evidently_report
            with st.spinner('Evaluating full held-out period...'):
                path = build_evidently_report()
            st.success(f'Report saved: {path.relative_to(ROOT)}')
            st.rerun()
        except Exception as exc:
            st.error(f'Evidently generation failed: {exc}')

st.divider()
st.caption('HUST Business Analytics Capstone #8 • Leakage-Safe Point & Quantile Forecasting • Scenario Capacity Planning')
