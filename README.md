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

Future Improvements
Integrate ARIMA / Prophet time-series models
Schedule automated retraining with new sales data

Author
Arun Nathi
📍 Dallas, TX
📧 nathiarun1999@gmail.com
🔗 LinkedIn Profile:



