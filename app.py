import sys
import io
import base64
import os
import numpy as np
import pandas as pd
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

MIN_FREEZE_RUN = 5
IQR_MULTIPLIER = 2.5
NIGHT_O3_THRESHOLD = 40
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6


def fix_sensor_freeze(df, col, min_run=5):
    values = df[col]
    group_id = (values != values.shift(1)).cumsum()
    run_length = values.groupby(group_id).transform("size")
    mask = run_length >= min_run
    df.loc[mask, col] = np.nan
    df = df.set_index("date")
    df[col] = df[col].interpolate(method="time")
    df = df.reset_index()
    return df


def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def get_pm25_subindex(x):
    if pd.isna(x):
        return 0
    elif x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    else:
        return 400 + (x - 250) * 100 / 130


def get_pm10_subindex(x):
    if pd.isna(x):
        return 0
    elif x <= 50:
        return x
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250) * 100 / 100
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    else:
        return 400 + (x - 430) * 100 / 80


def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delhi AQI Analytics Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; }
        header { text-align: center; margin-bottom: 20px; }
        .dashboard-layout { display: flex; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .console-panel { flex: 1; background-color: #1e1e1e; color: #39ff14; padding: 20px; border-radius: 8px; max-height: 850px; overflow-y: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
        .console-panel h3 { color: #ffffff; margin-top: 0; border-bottom: 1px solid #444; padding-bottom: 8px; }
        pre { font-family: 'Courier New', monospace; white-space: pre-wrap; font-size: 13px; line-height: 1.5; margin: 0; }
        .visual-panel { flex: 1.2; display: flex; flex-direction: column; gap: 20px; }
        .chart-card { background: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }
        .chart-card h3 { margin-top: 0; color: #333; border-bottom: 2px solid #eaeaea; padding-bottom: 5px; text-align: left; }
        .chart-card img { max-width: 100%; height: auto; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <header>
        <h1>Delhi AQI Prediction Pipeline System</h1>
        <p>Local Web Dashboard Monitoring Engine</p>
    </header>
    <div class="dashboard-layout">
        <div class="console-panel">
            <h3>System Terminal Outputs</h3>
            <pre>{{ output }}</pre>
        </div>
        <div class="visual-panel">
            <div class="chart-card">
                <h3>Feature Importance Matrix Analysis</h3>
                {% if chart1 %}
                    <img src="{{ chart1 }}" alt="Feature Importance Plot">
                {% else %}
                    <p style="color:red;">Chart 1 failed to build.</p>
                {% endif %}
            </div>
            <div class="chart-card">
                <h3>Random Forest Validation: Actual vs Predictions</h3>
                {% if chart2 %}
                    <img src="{{ chart2 }}" alt="Actual vs Predicted Plot">
                {% else %}
                    <p style="color:red;">Chart 2 failed to build.</p>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
"""


def run_pipeline():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    plot1_b64, plot2_b64 = "", ""
    try:
        print("==========================================================")
        print("                 DELHI AQI ML DATA PIPELINE               ")
        print("==========================================================")

        if not os.path.exists("delhi_aqi.csv"):
            print("\n[CRITICAL ERROR] Missing 'delhi_aqi.csv' file!")
            print(f"Current Directory Location: {os.getcwd()}")
            return sys.stdout.getvalue(), "", ""

        print("\n[1] DATASET LOADING")
        print("-> Using source dataset file: 'delhi_aqi.csv'")
        df = pd.read_csv("delhi_aqi.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        print(f"-> Original Raw Data Shape: {df.shape[0]} hourly rows and {df.shape[1]} columns.")

        print("\n[2] DATA PREPROCESSING & CLEANING")
        full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="1h")
        df = df.set_index("date").reindex(full_range)
        df.index.name = "date"
        df[numeric_cols] = df[numeric_cols].interpolate(method="time")


        df = df.reset_index()

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            df[col] = df[col].clip(upper=Q3 + 1.5 * (Q3 - Q1))

        for col in ["no", "o3"]:
            df = fix_sensor_freeze(df, col, MIN_FREEZE_RUN)

        df = df.set_index("date")
        df["hour"] = df.index.hour
        night_mask = ((df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)) & (
                    df["o3"] > NIGHT_O3_THRESHOLD)
        df.loc[night_mask, "o3"] = np.nan
        df["o3"] = df["o3"].interpolate(method="time")
        night_hours = (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
        df.loc[night_hours & (df["o3"] > NIGHT_O3_THRESHOLD), "o3"] = NIGHT_O3_THRESHOLD
        df = df.drop(columns="hour").reset_index()
        print("-> Completed missing hour interpolation, outlier mitigation, and nighttime ozone filtering.")

        print("\n[3] FEATURE ENGINEERING")
        df["hour"] = df["date"].dt.hour
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["season"] = df["month"].apply(get_season)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        pollutant_columns = ["pm2_5", "pm10", "no", "no2", "nh3", "co", "so2", "o3"]
        for col in pollutant_columns:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1)
                df[f"{col}_lag24"] = df[col].shift(24)

        df = df.iloc[24:].reset_index(drop=True)
        df["pm2_5_raw"] = df["pm2_5"]
        df["pm10_raw"] = df["pm10"]
        print("-> Engineered chronological variables, cyclic encodings, and historical 1h/24h lag variables.")

        print("\n[4] DATA SPLITTING & MODEL TRAINING")
        df["pm25_aqi"] = df["pm2_5_raw"].apply(get_pm25_subindex)
        df["pm10_aqi"] = df["pm10_raw"].apply(get_pm10_subindex)
        df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1).round(0)
        df = df.drop(columns=["pm25_aqi", "pm10_aqi", "pm2_5_raw", "pm10_raw"])

        X = df.drop(columns=["aqi", "pm2_5", "pm10", "date"], errors="ignore")
        y = df["aqi"]

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        total_rows = len(df)
        train_start_row, train_end_row = 0, split_idx - 1
        test_start_row, test_end_row = split_idx, total_rows - 1

        print(f"-> Total Rows Available after cleaning: {total_rows}")
        print(
            f"-> TRAINING ROWS USED : Row #{train_start_row} to Row #{train_end_row} ({len(X_train)} chronological rows)")
        print(
            f"-> TESTING ROWS USED  : Row #{test_start_row} to Row #{test_end_row} ({len(X_test)} chronological rows)")

        model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"-> Model Used: Random Forest Regressor")
        print(f"-> Performance Results on Test Set:")
        print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
        print(f"   - R-squared (R2) Score: {r2:.2f}")

        print("\n[5] EXPLANATION OF VISUAL GRAPH METRICS")
        print("-> GRAPH 1: Feature Importance Matrix")
        print(
            f"   - Generated using pattern metrics learned strictly from data rows #{train_start_row} to #{train_end_row}.")
        print("\n-> GRAPH 2: Actual vs Predicted AQI Scatter")
        print(f"   - Maps exactly {len(y_test)} rows belonging to rows #{test_start_row} to #{test_end_row}.")
        print("==========================================================")


        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(
            by="Importance").tail(15)

        fig1 = plt.figure(figsize=(10, 5))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
        plt.xlabel("Importance Score")
        plt.title("Top 15 Random Forest Feature Importances")
        plot1_b64 = fig_to_base64(fig1)

        fig2 = plt.figure(figsize=(8, 5))
        plt.scatter(y_test, predictions, color="steelblue", alpha=0.5, edgecolors="black", s=25)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title(f"Actual vs Predicted AQI (Data Rows {test_start_row} to {test_end_row})")
        plt.grid(True, alpha=0.2)
        plot2_b64 = fig_to_base64(fig2)

    except Exception as e:
        print(f"\n[CRITICAL RUNTIME ERROR]: {str(e)}")
    finally:
        terminal_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    return terminal_output, plot1_b64, plot2_b64


@app.route("/")
def home():
    output_text, graph1, graph2 = run_pipeline()
    return render_template_string(HTML_TEMPLATE, output=output_text, chart1=graph1, chart2=graph2)


if __name__ == "__main__":
    app.run(debug=True, port=8888)