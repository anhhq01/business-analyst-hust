"""Generate an Evidently regression HTML report and/or publish to the local Evidently UI workspace."""
from __future__ import annotations

import pandas as pd

from config import EVIDENTLY_HTML, EVIDENTLY_PROJECT_NAME, EVIDENTLY_WORKSPACE, PREDICTIONS


def build_evidently_report(predictions_path=PREDICTIONS, output_path=EVIDENTLY_HTML):
    """Generate a self-contained Evidently HTML regression & data summary report."""
    from evidently import Dataset, DataDefinition, Report
    from evidently.core.datasets import Regression
    from evidently.presets import DataSummaryPreset, RegressionPreset

    df = pd.read_csv(predictions_path, parse_dates=['timestamp'])
    eval_df = df.rename(columns={
        'system_load': 'target',
        'prediction_lgbm': 'prediction',
    }).copy()

    definition = DataDefinition(
        timestamp='timestamp',
        numerical_columns=['target', 'prediction', 'capacity_threshold'],
        categorical_columns=['actual_peak', 'predicted_peak'],
        regression=[Regression(target='target', prediction='prediction')],
    )
    dataset = Dataset.from_pandas(eval_df, data_definition=definition)
    report = Report([RegressionPreset(), DataSummaryPreset()])
    snapshot = report.run(dataset, None, name='tetouan_1h_forecast_test')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    return output_path


def configure_evidently_dashboard(project):
    """Add an idempotent overview and data profile to the Evidently dashboard."""
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import DashboardPanelPlot

    dashboard = project.dashboard
    dashboard_model = dashboard.model()
    for tab in dashboard_model.tabs:
        tab.panels = list(dict.fromkeys(tab.panels))
    dashboard._workspace.save_dashboard(project.id, dashboard_model)

    for obsolete_title in (
        'MAE across evaluation runs',
        'Dataset size',
        'Actual and forecast load profile',
        'Peak label counts',
    ):
        model = dashboard.model()
        panel = next((item for item in model.panels if item.title == obsolete_title), None)
        if panel is not None:
            owner = next((tab.title for tab in model.tabs if panel.id in tab.panels), None)
            if owner is not None:
                dashboard.delete_panel(obsolete_title, owner)

    panels = [
        (
            'Model evaluation',
            DashboardPanelPlot(
                title='Regression quality',
                subtitle='LightGBM quality on the fixed held-out test period.',
                size='full',
                values=[
                    PanelMetric(legend='MAE mean', metric='MAE', metric_labels={'value_type': 'mean'}),
                    PanelMetric(legend='MAE std', metric='MAE', metric_labels={'value_type': 'std'}),
                    PanelMetric(legend='RMSE', metric='RMSE'),
                    PanelMetric(legend='MAPE mean (%)', metric='MAPE', metric_labels={'value_type': 'mean'}),
                    PanelMetric(legend='MAPE std', metric='MAPE', metric_labels={'value_type': 'std'}),
                    PanelMetric(legend='R2', metric='R2Score'),
                    PanelMetric(legend='Mean error', metric='MeanError', metric_labels={'value_type': 'mean'}),
                    PanelMetric(legend='Mean error std', metric='MeanError', metric_labels={'value_type': 'std'}),
                    PanelMetric(legend='Maximum absolute error', metric='AbsMaxError'),
                ],
                plot_params={'plot_type': 'counter', 'aggregation': 'last'},
            ),
        ),
        (
            'Data profile',
            DashboardPanelPlot(
                title='Data quality summary',
                subtitle='Completeness, duplication, and field-type checks for the evaluated dataset.',
                size='full',
                values=[
                    PanelMetric(legend='Rows', metric='RowCount'),
                    PanelMetric(legend='Numeric columns', metric='ColumnCount', metric_labels={'column_type': 'num'}),
                    PanelMetric(legend='Categorical columns', metric='ColumnCount', metric_labels={'column_type': 'cat'}),
                    PanelMetric(legend='Missing cells', metric='DatasetMissingValueCount', metric_labels={'value_type': 'count'}),
                    PanelMetric(legend='Missing share', metric='DatasetMissingValueCount', metric_labels={'value_type': 'share'}),
                    PanelMetric(legend='Duplicate rows', metric='DuplicatedRowCount'),
                    PanelMetric(legend='Duplicate columns', metric='DuplicatedColumnsCount'),
                    PanelMetric(legend='Empty columns', metric='EmptyColumnsCount'),
                    PanelMetric(legend='Constant columns', metric='ConstantColumnsCount'),
                ],
                plot_params={'plot_type': 'counter', 'aggregation': 'last'},
            ),
        ),
        (
            'Data profile',
            DashboardPanelPlot(
                title='Actual vs forecast load range',
                subtitle='Compare the actual and predicted load distribution summaries, in source units.',
                size='full',
                values=[
                    PanelMetric(legend='Actual min', metric='MinValue', metric_labels={'column': 'target'}),
                    PanelMetric(legend='Actual mean', metric='MeanValue', metric_labels={'column': 'target'}),
                    PanelMetric(legend='Actual max', metric='MaxValue', metric_labels={'column': 'target'}),
                    PanelMetric(legend='Forecast min', metric='MinValue', metric_labels={'column': 'prediction'}),
                    PanelMetric(legend='Forecast mean', metric='MeanValue', metric_labels={'column': 'prediction'}),
                    PanelMetric(legend='Forecast max', metric='MaxValue', metric_labels={'column': 'prediction'}),
                ],
                plot_params={'plot_type': 'bar', 'is_stacked': False},
            ),
        ),
        (
            'Data profile',
            DashboardPanelPlot(
                title='Peak breach comparison',
                subtitle='Hours above and below the training-derived scenario threshold.',
                size='half',
                values=[
                    PanelMetric(legend='Actual breach hours', metric='UniqueValueCount',
                                metric_labels={'column': 'actual_peak', 'label': '1', 'value_type': 'count'}),
                    PanelMetric(legend='Predicted breach hours', metric='UniqueValueCount',
                                metric_labels={'column': 'predicted_peak', 'label': '1', 'value_type': 'count'}),
                    PanelMetric(legend='Actual below threshold', metric='UniqueValueCount',
                                metric_labels={'column': 'actual_peak', 'label': '0', 'value_type': 'count'}),
                    PanelMetric(legend='Predicted below threshold', metric='UniqueValueCount',
                                metric_labels={'column': 'predicted_peak', 'label': '0', 'value_type': 'count'}),
                ],
                plot_params={'plot_type': 'bar', 'is_stacked': False},
            ),
        ),
    ]

    existing_titles = {panel.title for panel in dashboard.model().panels}
    for tab, panel in panels:
        if panel.title not in existing_titles:
            dashboard.add_panel(panel, tab=tab)
            existing_titles.add(panel.title)

    dashboard_model = dashboard.model()
    for tab in dashboard_model.tabs:
        tab.panels = list(dict.fromkeys(tab.panels))
    dashboard._workspace.save_dashboard(project.id, dashboard_model)


def publish_evidently_run(predictions_path=PREDICTIONS):
    """Evaluate the saved test predictions and add the snapshot to Evidently UI."""
    from evidently import Dataset, DataDefinition, Report
    from evidently.core.datasets import Regression
    from evidently.presets import DataSummaryPreset, RegressionPreset
    from evidently.ui.workspace import Workspace

    df = pd.read_csv(predictions_path, parse_dates=['timestamp'])
    eval_df = df.rename(columns={
        'system_load': 'target',
        'prediction_lgbm': 'prediction',
    }).copy()

    definition = DataDefinition(
        timestamp='timestamp',
        numerical_columns=['target', 'prediction', 'capacity_threshold'],
        categorical_columns=['actual_peak', 'predicted_peak'],
        regression=[Regression(target='target', prediction='prediction')],
    )
    dataset = Dataset.from_pandas(eval_df, data_definition=definition)
    report = Report([RegressionPreset(), DataSummaryPreset()])
    snapshot = report.run(dataset, None, name='tetouan_1h_forecast_test')

    workspace = Workspace.create(str(EVIDENTLY_WORKSPACE))
    matching_projects = workspace.search_project(EVIDENTLY_PROJECT_NAME)
    if matching_projects:
        project = matching_projects[0]
    else:
        project = workspace.create_project(EVIDENTLY_PROJECT_NAME)
        project.description = (
            'Historical one-hour-ahead Tetouan load regression evaluation. '
            'Each run evaluates the fixed held-out test predictions; this is not live monitoring.'
        )
        project.save()

    configure_evidently_dashboard(project)
    workspace.add_run(project.id, snapshot, include_data=True)
    return project


if __name__ == '__main__':
    html_path = build_evidently_report()
    print(f'Saved Evidently HTML report: {html_path}')
    project = publish_evidently_run()
    print(f'Added evaluation run to Evidently project: {project.name} ({project.id})')
    print(f'Workspace: {EVIDENTLY_WORKSPACE}')
    print(
        'PowerShell: $env:DO_NOT_TRACK="1"; '
        f'evidently ui --workspace "{EVIDENTLY_WORKSPACE}" --port 8000'
    )
