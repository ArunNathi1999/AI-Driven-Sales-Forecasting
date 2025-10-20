# 🤖 AI-Driven Sales Forecasting

An end-to-end Machine Learning project that predicts future sales trends based on historical data using regression models.  
This project helps businesses forecast demand, plan inventory, and make data-driven decisions.

---

## 📊 Key Features
- 🧹 **Data Preprocessing** – Cleans and structures raw sales data  
- ⚙️ **Model Training** – Builds and tunes ML models (`RandomForestRegressor`, `DecisionTreeRegressor`)  
- 🔮 **Forecasting** – Predicts upcoming 4-week sales for each product  
- 📈 **Visualization Dashboard** – Compares historical vs predicted sales trends  
- 💾 **Model Persistence** – Saves trained model using `joblib` for reuse

---

## 🗂️ Project Structure
AI-Driven-Sales-Forecasting/
│
├── data/ # raw_sales.csv, cleaned_sales.csv
├── models/ # saved ML model (sales_forecast_model.pkl)
├── reports/ # forecast results and charts
├── scripts/ # main python scripts
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── forecast_future.py
│
├── notebooks/ # Jupyter notebooks (optional)
├── requirements.txt # dependencies
└── README.md


---

## 🧠 Tech Stack
- **Python 3.12**
- **Pandas**, **NumPy**, **Matplotlib**, **Scikit-learn**
- **Joblib** for model saving
- **Git + GitHub** for version control

---

## 🧪 How to Run the Project

1️⃣ **Clone the repository**
```bash
git clone https://github.com/ArunNathi1999/AI-Driven-Sales-Forecasting.git
cd AI-Driven-Sales-Forecasting

Setup Environment
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run scripts
python scripts/data_preprocessing.py
python scripts/model_training.py!
python scripts/forecast_future.py

Results are Uploaded in files section

Results

RMSE: ~15.4
R² Score: -0.11 (initial baseline)
Generates weekly forecasts for multiple products



Here is the coding part
## 🧩 Quickstart: Run Everything in One Cell (Python)

```python
# Quickstart cell: load data & model, make plots, generate 4-week forecast
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib

# Project paths (relative so this works for anyone who clones the repo)
PROJECT = Path(".").resolve()
DATA    = PROJECT / "data"
MODELS  = PROJECT / "models"
REPORTS = PROJECT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# 1) Load cleaned data
df = pd.read_csv(DATA / "cleaned_sales.csv", parse_dates=["date"])
print("✅ Data loaded:", df.shape)

# 2) Load trained model
model_path = MODELS / "sales_forecast_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded from:", model_path)

# Helper to save figures
def savefig(name, dpi=160):
    out = REPORTS / name
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print("Saved →", out.resolve())

# 3) Visualizations
## Historical lines
plt.figure(figsize=(9, 4.6))
for prod, g in df.groupby("product"):
    g = g.sort_values("date")
    plt.plot(g["date"], g["units_sold"], label=prod, linewidth=1.5)
plt.title("Weekly Units Sold by Product")
plt.xlabel("Date"); plt.ylabel("Units Sold"); plt.legend()
savefig("hist_units.png"); plt.show()

## 4-week moving average
plt.figure(figsize=(9, 4.6))
for prod, g in df.groupby("product"):
    g = g.sort_values("date")
    ma = g["units_sold"].rolling(4, min_periods=1).mean()
    plt.plot(g["date"], ma, label=f"{prod} (4-wk MA)", linewidth=2)
plt.title("Smoothed Trend (4-Week Moving Average)")
plt.xlabel("Date"); plt.ylabel("Units (MA)"); plt.legend()
savefig("trend_ma.png"); plt.show()

## Feature importance (RandomForest only)
if hasattr(model, "feature_importances_"):
    import numpy as np
    features = ["lag_1", "lag_2", "lag_3", "promo_flag"]
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(np.array(features)[order], importances[order])
    plt.title("Model Feature Importance")
    plt.ylabel("Importance")
    savefig("feature_importance.png"); plt.show()
else:
    print("Model has no feature_importances_ — skipping bar chart.")

# 4) 4-week recursive forecast per product
df = df.copy()
if not np.issubdtype(df["date"].dtype, np.datetime64):
    df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["product", "date"])

horizon = 4
rows = []

for prod, g in df.groupby("product"):
    g = g.sort_values("date").reset_index(drop=True)
    last_date = g["date"].iloc[-1]
    lags = list(g["units_sold"].tail(3).astype(float).values)
    while len(lags) < 3:  # fallback if a series is short
        lags.insert(0, lags[-1])

    for step in range(1, horizon + 1):
        X = pd.DataFrame([{
            "lag_1": lags[-1],
            "lag_2": lags[-2],
            "lag_3": lags[-3],
            "promo_flag": 0,  # set 1 to simulate promo
        }])
        yhat = float(model.predict(X)[0])
        rows.append({
            "product": prod,
            "date": last_date + pd.Timedelta(weeks=step),
            "forecast_units": yhat
        })
        lags.append(yhat)

fc_4w = pd.DataFrame(rows)
csv_out = REPORTS / "four_week_forecast.csv"
fc_4w.to_csv(csv_out, index=False)
print("✅ Forecast saved →", csv_out.resolve())

# Overlay plot (history + forecast)
plt.figure(figsize=(10.5, 5.2))
for prod, g in df.groupby("product"):
    plt.plot(g["date"], g["units_sold"], label=f"{prod} — history", linewidth=1.4)
for prod, g in fc_4w.groupby("product"):
    g = g.sort_values("date")
    plt.plot(g["date"], g["forecast_units"], "--", label=f"{prod} — forecast (4w)", linewidth=2)
plt.title("Historical Units vs 4-Week Forecast")
plt.xlabel("Date"); plt.ylabel("Units")
plt.legend(ncol=2, fontsize=9)
savefig("forecast_dashboard.png"); plt.show()

Future Improvements
Integrate ARIMA / Prophet time-series models
Schedule automated retraining with new sales data

Author
Arun Nathi
📍 Dallas, TX
📧 nathiarun1999@gmail.com
🔗 LinkedIn Profile:



