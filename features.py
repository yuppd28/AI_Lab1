# src/features.py
import pandas as pd

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Категоріальні колонки для one-hot кодування
    cat_cols = [
        "airline", "flight", "source_city", "departure_time",
        "stops", "arrival_time", "destination_city", "class"
    ]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


