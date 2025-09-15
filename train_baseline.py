# src/train_baseline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.data import load_data
from src.features import preprocess
from src.eval import evaluate

def main():
    os.makedirs("reports/tables", exist_ok=True)
    df = load_data()
    df = preprocess(df)

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    metrics.to_csv("reports/tables/baseline_metrics.csv", index=False)

    # ⬇⬇⬇ додано: збереження прогнозів для Dash
    forecast = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    forecast.to_csv("reports/tables/baseline_forecast.csv", index=False)

    print(metrics)

if __name__ == "__main__":
    main()
