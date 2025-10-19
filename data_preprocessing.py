
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def main():
    raw = pd.read_csv(DATA_DIR / "raw_sales.csv", parse_dates=["date"])
    df = raw.copy()

    # Basic cleaning
    df = df.dropna()
    df = df[df["units_sold"] >= 0]

    # Feature examples
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    # Aggregate to weekly product-level (example)
    weekly = (
        df.groupby(["date", "product"], as_index=False)
        .agg({"units_sold": "sum", "revenue": "sum", "promo_flag": "max"})
        .sort_values("date")
    )

    out_path = DATA_DIR / "cleaned_sales.csv"
    weekly.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path.resolve()} (rows={len(weekly)})")

if __name__ == "__main__":
    main()
