
# 🤖 AI-Driven Sales Forecasting (Starter)

This starter contains a clean folder structure, sample `raw_sales.csv`, and minimal Python scripts
so you can push a working project to GitHub immediately.

## Structure
- `data/` — sample raw dataset and your future cleaned dataset
- `notebooks/` — EDA, feature engineering, modeling notebooks
- `scripts/` — Python scripts for preprocessing, training, and forecasting
- `models/` — saved models (pickle/joblib)
- `reports/` — charts, metrics, screenshots

## Quickstart
1. Create a virtual env and install dependencies:
```
pip install -r requirements.txt
```
2. Run preprocessing to create `cleaned_sales.csv`:
```
python scripts/data_preprocessing.py
```
3. Train a model and save metrics and `models/sales_forecast_model.pkl`:
```
python scripts/model_training.py
```
4. Generate a simple future forecast plot and CSV:
```
python scripts/forecast_future.py
```
