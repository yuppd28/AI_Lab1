# src/data.py
import pandas as pd
import os

def load_data():
    # автоматично шукає data/flight_price.csv у корені проєкту
    base_dir = os.path.dirname(os.path.dirname(__file__))  # піднімаємось на рівень вище від src
    path = os.path.join(base_dir, "data", "flight_price.csv")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
