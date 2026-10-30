from preprocessing import build_model_table
from modeling import fit_baseline

if __name__ == '__main__':
    _, quality = build_model_table()
    _, metrics, _ = fit_baseline()
    print('Pipeline complete.')
    print('Data quality:', quality)
    print('Model metrics:', metrics)
    print('Run `python src/visualization.py` to publish the test evaluation to Evidently UI.')
    print('The script prints the PowerShell command to launch Evidently UI with this workspace.')
