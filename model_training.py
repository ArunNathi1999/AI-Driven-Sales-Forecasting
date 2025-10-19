
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_DIR / "cleaned_sales.csv", parse_dates=["date"])

    # Create lag features for time-series style supervised learning
    df = df.sort_values(["product", "date"])
    df["lag_1"] = df.groupby("product")["units_sold"].shift(1)
    df["lag_2"] = df.groupby("product")["units_sold"].shift(2)
    df["lag_3"] = df.groupby("product")["units_sold"].shift(3)
    df = df.dropna().reset_index(drop=True)

    features = ["lag_1", "lag_2", "lag_3", "promo_flag"]
    X = df[features]
    y = df["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.3f} | R^2: {r2:.3f}")

    out_path = MODEL_DIR / "sales_forecast_model.pkl"
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path.resolve()}")

if __name__ == "__main__":
    main()
