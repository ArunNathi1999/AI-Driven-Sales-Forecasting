
import pandas as pd
import joblib
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_DIR / "cleaned_sales.csv", parse_dates=["date"])
    df = df.sort_values(["product", "date"])

    # Load trained model
    model = joblib.load(MODEL_DIR / "sales_forecast_model.pkl")

    # Build latest feature rows for each product
    forecasts = []
    for prod, g in df.groupby("product"):
        g = g.sort_values("date").tail(3)  # last 3 rows for lags
        if len(g) < 3:
            continue
        row = {
            "product": prod,
            "lag_1": g.iloc[-1]["units_sold"],
            "lag_2": g.iloc[-2]["units_sold"],
            "lag_3": g.iloc[-3]["units_sold"],
            "promo_flag": 0,
        }
        X = pd.DataFrame([row])[["lag_1", "lag_2", "lag_3", "promo_flag"]]
        yhat = model.predict(X)[0]
        forecasts.append({"product": prod, "next_period_units_forecast": float(yhat)})

    out_csv = REPORT_DIR / "next_period_forecast.csv"
    pd.DataFrame(forecasts).to_csv(out_csv, index=False)
    print(f"Saved simple next-period forecast to {out_csv.resolve()}")

if __name__ == "__main__":
    main()
