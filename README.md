# Employee Attrition Prediction & Analysis (NHA-5-153)

Predicting employee turnover so HR can act early to keep people. The project runs through the full
data-science lifecycle: data, EDA, feature engineering, modelling, MLOps / deployment, and final presentation.

## Milestone 1: Data Collection, Exploration & Preprocessing

| Task | Owner | Deliverable |
|---|---|---|
| T1 Data cleaning | Raghad | [`depi_Project(data_cleaning).ipynb`](depi_Project(data_cleaning).ipynb) → `employee_attriton_data_clean.csv` |
| T2 Transformation (scaling, encoding, interaction features) | Sama | same notebook, *Transformation* section → `employee_attrition_transformed.csv` |
| T3 Data visualization | Youssef | [`notebooks/03_eda_visualization.ipynb`](notebooks/03_eda_visualization.ipynb), `src/eda_pipeline.py`, `reports/figures/` |
| T4 EDA report | Youssef | [`reports/EDA_Report.md`](reports/EDA_Report.md) |

## Repository layout

```
├── raw_employee_attrition_data_(inprogress).csv   # raw data (50,000 rows)
├── employee_attriton_data_clean.csv               # T1 output - cleaned (49,000 rows)
├── employee_attrition_transformed.csv             # T2 output - scaled + encoded + engineered
├── depi_Project(data_cleaning).ipynb              # T1 + T2 notebook (Colab)
├── notebooks/
│   └── 03_eda_visualization.ipynb                 # T3 EDA visual notebook
├── src/
│   └── eda_pipeline.py                            # reproducible figures + summary tables
├── reports/
│   ├── EDA_Report.md                              # T4 report
│   ├── figures/        *.png                      # static charts
│   ├── interactive/    *.html                     # Plotly charts (open in a browser)
│   └── tables/         *.csv                      # summary statistics & statistical tests
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python src/eda_pipeline.py                    # regenerates reports/figures, tables, interactive
jupyter notebook notebooks/03_eda_visualization.ipynb
```

Or open the notebooks in Google Colab with the badge at the top of each one.
