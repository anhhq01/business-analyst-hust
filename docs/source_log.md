# Source log

- Dataset: Power Consumption of Tetouan City
- Host: [UCI Machine Learning Repository, dataset 849](https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity)
- DOI: 10.24432/C5B034
- Creators: Abdulwahed Salam; Abdelaaziz El Hibaoui
- Citation: Salam, A. & El Hibaoui, A. (2018). *Power Consumption of Tetouan City* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B034.
- License/terms: CC BY 4.0, as listed by UCI; attribution is required. Check the linked UCI record for the license text and terms before redistribution.
- UCI record checked: 2026-10-30.
- Original CSV download date: not recorded in the project. The local file's filesystem creation time is not reliable evidence of its original download date.
- Local file creation timestamp observed: 2026-10-30 09:19 (machine local time); this may reflect copying into the workspace and is not represented as the download date.
- Supplied project file: `data/raw/Tetuan City power consumption.csv`
- Raw rows in supplied CSV: 52,416
- Observed period in supplied CSV: 2017-01-01 00:00 through 2017-12-30 23:50
- Raw cadence: 10 minutes
- Catalog note: UCI lists 52,417 instances; the supplied local CSV contains 52,416 data rows. The analysis uses the supplied file as-is and records its actual row count rather than adding or synthesizing a row.

## Filtering, joining, and transformation decisions

1. Read the supplied CSV; normalize whitespace in column names; parse timestamps strictly and sort ascending.
2. Check required columns, invalid/duplicate timestamps, missing cells, and non-10-minute time gaps. Invalid or duplicate timestamps stop the pipeline. Gap counts are recorded; the validation function does not currently reject a file solely because it contains a gap.
3. No external tables are joined. No raw value imputation or row-level outlier filtering is performed.
4. Resample weather and three zone measurements to hourly means. Keep only hours with all six expected 10-minute observations; incomplete hours are dropped rather than imputed.
5. Define `system_load` as the sum of the three hourly zone means. This is a combined load index; it is not converted to energy because source units/integration rules are not established.
6. Create lag and rolling features. Drop the first rows that lack the required 168-hour history (and any other feature warm-up values); do not fill them.
7. Use chronological blocks: training before validation; validation is used for early stopping; the final 30 days are held out for evaluation. No random sampling or joins are used.

The exact resulting row counts and time boundaries are written to `data/processed/data_quality_summary.json` and `reports/metrics.json` when the pipeline runs.
