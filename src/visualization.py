"""Generate an Evidently regression report for live/demo use.

Tested against the Evidently 0.7 API shape documented in 2026.
The HTML report is the zero-infrastructure demo. For a hosted local UI,
run `evidently ui` in a second terminal and optionally upload/store runs.
"""
from __future__ import annotations
import pandas as pd

from config import PREDICTIONS, EVIDENTLY_HTML


def build_evidently_report(predictions_path=PREDICTIONS, output_path=EVIDENTLY_HTML):
    from evidently import Dataset, DataDefinition, Report
    from evidently.core.datasets import Regression
    from evidently.presets import RegressionPreset, DataSummaryPreset

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


if __name__ == '__main__':
    path = build_evidently_report()
    print(f"Saved Evidently HTML report: {path}")
