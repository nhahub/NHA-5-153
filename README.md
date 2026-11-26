# Employee Attrition Prediction & Analysis (NHA-5-153)

Predicting employee turnover so HR can act early to retain talent. The project covers the full
data-science lifecycle: data cleaning, EDA, statistical analysis, feature engineering, model development,
and production MLOps (MLflow, DVC, FastAPI, Streamlit, Docker, monitoring, and automated retraining).

## Milestones & Deliverables

### Milestone 1: Data Collection, Exploration & Preprocessing
| Task | Owner | Deliverable |
|---|---|---|
| T1 Data cleaning | Raghad | [`depi_Project(data_cleaning).ipynb`](depi_Project(data_cleaning).ipynb) → `employee_attriton_data_clean.csv` |
| T2 Transformation | Sama | same notebook → `employee_attrition_transformed.csv` |
| T3 Data visualization | Youssef | [`notebooks/03_eda_visualization.ipynb`](notebooks/03_eda_visualization.ipynb), `src/eda_pipeline.py`, `reports/figures/` |
| T4 EDA report | Youssef | [`reports/EDA_Report.md`](reports/EDA_Report.md) |

### Milestone 2: Advanced Data Analysis & Feature Engineering
| Task | Deliverable |
|---|---|
| T1 Statistical analysis & hypothesis testing | [`notebooks/04_statistical_analysis.ipynb`](notebooks/04_statistical_analysis.ipynb), `src/statistical_analysis.py`, [`reports/Data_Analysis_Report.md`](reports/Data_Analysis_Report.md) |
| T2 Feature selection & importance | [`notebooks/05_feature_selection.ipynb`](notebooks/05_feature_selection.ipynb) |
| T3 Feature engineering & transformation | [`notebooks/Feature Engineering & Transformation.ipynb`](notebooks/Feature%20Engineering%20%26%20Transformation.ipynb) → `transformed_hr_dataset.csv` |
| T4 Advanced visualizations | [`notebooks/Advanced_Visualization_&_Interactive_Dashboards.ipynb`](notebooks/Advanced_Visualization_%26_Interactive_Dashboards.ipynb) |

### Milestone 3: Machine Learning Model Development & Optimization
| Deliverable | Location |
|---|---|
| Model comparison, SMOTE, tuning & evaluation report | [`Milestone3_Model_Development.ipynb`](Milestone3_Model_Development.ipynb) |

### Milestone 4: MLOps, Deployment & Monitoring
| Task | Deliverable |
|---|---|
| T1 MLOps (MLflow + DVC) | [`src/attrition/train.py`](src/attrition/train.py), [`params.yaml`](params.yaml), [`dvc.yaml`](dvc.yaml), [`models/`](models/) |
| T2 Deployment (FastAPI + Streamlit + Docker) | [`src/attrition/api.py`](src/attrition/api.py), [`src/attrition/dashboard.py`](src/attrition/dashboard.py), [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml) |
| T3 Model monitoring & alerting | [`src/attrition/monitoring/`](src/attrition/monitoring/), [`.github/workflows/monitor.yml`](.github/workflows/monitor.yml) |
| T4 Retraining strategy & rollback | [`src/attrition/retrain.py`](src/attrition/retrain.py), [`docs/Monitoring_and_Retraining.md`](docs/Monitoring_and_Retraining.md) |
| MLOps Report | [`reports/MLOps_Report.md`](reports/MLOps_Report.md) |

## Quickstart (Milestone 4 Stack)

### Local (Python 3.10+)
```bash
pip install -e ".[dev]"

# 1. Train & register champion model in MLflow
python -m attrition.train

# 2. Start the real-time REST API (Swagger UI at http://localhost:8000/docs)
uvicorn attrition.api:app --host 0.0.0.0 --port 8000

# 3. Launch the HR Streamlit dashboard (http://localhost:8501)
streamlit run src/attrition/dashboard.py

# 4. Run monitoring / simulate drift / test retraining
python -m attrition.monitoring.simulate --scenario covariate_shift
python -m attrition.monitoring.monitor --window-hours 168
python -m attrition.retrain --dry-run

# 5. Run the automated test suite
pytest -q
```

### Docker Compose (API + Dashboard + Monitor + MLflow UI)
```bash
docker compose up --build
```
- **API & Swagger UI:** http://localhost:8000/docs
- **HR Dashboard:** http://localhost:8501
- **MLflow UI:** http://localhost:5000
