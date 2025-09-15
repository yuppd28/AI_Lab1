# src/train_rf.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.data import load_data
from src.features import preprocess
from src.eval import evaluate

def main(n_estimators=200, max_depth=None):
    os.makedirs("reports/tables", exist_ok=True)
    df = load_data()
    df = preprocess(df)

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    metrics.to_csv("reports/tables/rf_metrics.csv", index=False)

    forecast = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    forecast.to_csv("reports/tables/rf_forecast.csv", index=False)

    print("RandomForest metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
