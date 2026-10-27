from preprocessing import build_model_table
from modeling import fit_baseline

if __name__ == '__main__':
    _, quality = build_model_table()
    _, metrics, _ = fit_baseline()
    print('Pipeline complete.')
    print('Data quality:', quality)
    print('Model metrics:', metrics)
    print('Run `python src/visualization.py` after installing Evidently to create the HTML report.')
