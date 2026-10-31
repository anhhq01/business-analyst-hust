from preprocessing import build_model_table
from modeling import fit_baseline
from visualization import build_evidently_report

if __name__ == '__main__':
    _, quality = build_model_table()
    _, metrics, _ = fit_baseline()
    try:
        evidently_path = build_evidently_report()
        print(f'Evidently HTML report saved: {evidently_path}')
    except Exception as exc:
        print(f'Warning: Evidently report skipped ({exc})')
    print('Pipeline complete.')
    print('Data quality:', quality)
    print('Primary LightGBM metrics:', metrics['lightgbm'])
    print('Probabilistic metrics (calibrated P10-P90):', metrics['lightgbm_probabilistic_calibrated'])
    print('Rolling backtest summary:', metrics.get('rolling_backtest_aggregate'))
    print('Run `python src/visualization.py` to also publish the test evaluation to the Evidently UI workspace.')
