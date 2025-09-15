import os
import pandas as pd
from dash import Dash, dcc, html, dash_table
import plotly.graph_objects as go

app = Dash(__name__)
app.title = "Flight Price Forecasting"

def load_csv_safe(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def overlay_figure():
    figs = []
    actual = None
    for name, path in [
        ("Baseline", "reports/tables/baseline_forecast.csv"),
        ("RandomForest", "reports/tables/rf_forecast.csv"),
        ("XGBoost", "reports/tables/xgb_forecast.csv"),
    ]:
        df = load_csv_safe(path)
        if df is None:
            continue
        if actual is None:
            actual = df
        figs.append((name, df))

    fig = go.Figure()
    if actual is not None:
        fig.add_trace(go.Scatter(y=actual["y_true"], name="Actual", mode="lines"))
    for name, dfp in figs:
        fig.add_trace(go.Scatter(y=dfp["y_pred"], name=name, mode="lines"))
    fig.update_layout(title="Overlay: Actual vs Forecasts", legend=dict(orientation="h"))
    return fig

def side_by_side():
    panels = []
    for name, path in [
        ("Baseline", "reports/tables/baseline_forecast.csv"),
        ("RandomForest", "reports/tables/rf_forecast.csv"),
        ("XGBoost", "reports/tables/xgb_forecast.csv"),
    ]:
        df = load_csv_safe(path)
        if df is None:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["y_true"], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(y=df["y_pred"], name=name, mode="lines"))
        fig.update_layout(title=name)
        panels.append(html.Div(className="card", children=[dcc.Graph(figure=fig)]))
    return panels

def metrics_table():
    frames = []
    for name, path in [
        ("Baseline", "reports/tables/baseline_metrics.csv"),
        ("RandomForest", "reports/tables/rf_metrics.csv"),
        ("XGBoost", "reports/tables/xgb_metrics.csv"),
    ]:
        df = load_csv_safe(path)
        if df is None:
            continue
        df.insert(0, "Model", name)
        frames.append(df)
    if not frames:
        return html.Div("Немає метрик. Спочатку запусти тренування.", className="card")
    tab = pd.concat(frames, ignore_index=True)
    return dash_table.DataTable(
        data=tab.round(3).to_dict("records"),
        columns=[{"name": c, "id": c} for c in tab.columns],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px"},
        page_size=10
    )

app.layout = html.Div([
    html.H2("Flight Price Prediction — Dashboard"),
    html.H3("Overlay"),
    html.Div(className="card", children=[dcc.Graph(figure=overlay_figure())]),
    html.H3("Side-by-side"),
    html.Div(children=side_by_side()),
    html.H3("Metrics"),
    html.Div(className="card", children=[metrics_table()]),
])

if __name__ == "__main__":
    app.run(debug=True)
