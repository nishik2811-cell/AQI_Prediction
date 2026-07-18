# Delhi AQI Prediction

A Random Forest pipeline that predicts Delhi's Air Quality Index (AQI) from two years of hourly pollutant readings (CO, NO, NO₂, SO₂, O₃, PM2.5, PM10, NH₃), paired with a static, data-driven dashboard that walks through the cleaning, feature engineering, and model results.

**[View the live dashboard →](index.html)**

## What's in here

| File | Purpose |
|---|---|
| `main_changes.py` | **The pipeline to run.** Final, corrected version — see [Pipeline versions](#pipeline-versions) below. |
| `main.py`, `main_new.py` | Earlier iterations of the pipeline, kept for reference. |
| `delhi_aqi.csv` | Raw hourly pollutant readings. |
| `delhi_aqi_final.csv` | Output after anomaly fixing / cleaning. |
| `delhi_aqi_feature_engineered.csv` | Output after feature engineering, ready for model training. |
| `index.html` | Static dashboard presenting the pipeline stages, data patterns, and model performance. |
| `data.js` | Precomputed stats (row counts, MAE, R², hourly/monthly AQI patterns, feature importances) consumed by `index.html`. |

## Pipeline versions

There are three copies of the pipeline in the repo, representing its evolution. **`main_changes.py` is the one to use** - it fixes three bugs present in the earlier versions:

1. **Sensor freeze detection undercounted run lengths.** The earlier `value == shift(1)` check never flags the first value in a frozen run, so it was never replaced. `main_changes.py` groups consecutive equal values directly, capturing the full run.
2. **Rolling-window features leaked the current hour.** Pandas' `.rolling()` is trailing-inclusive by default, so `pm2_5_rolling3` / `pm10_rolling3` etc. included the very row the AQI target was computed from. Fixed by shifting one step before rolling, so windows only look at strictly past hours.
3. **AQI was computed from scaled/engineered values instead of raw µg/m³ readings.** The CPCB sub-index formulas (e.g. `x <= 30`, `x <= 60`) assume real-world units, so raw `pm2_5`/`pm10` are preserved separately and used for the AQI target.

The pipeline also uses a **chronological train/test split** (`shuffle=False`) rather than a random split, since lag and rolling features make neighboring hours look artificially similar — a random split would leak information from the test set into training.

## Pipeline stages (`main_changes.py`)

1. **Load & sanity checks** — missing values, duplicate timestamps, negative readings.
2. **Time-gap filling** — reindexes to a continuous hourly range and interpolates.
3. **Outlier capping** — IQR-based capping, then a wider 2.5× IQR ceiling pass.
4. **Sensor freeze correction** — replaces runs of ≥5 identical consecutive values (stuck sensors) with interpolated values, run on `no` and `o3`.
5. **Nighttime O₃ correction** — ozone should fall after dark (it forms from sunlight); implausibly high overnight O₃ readings are interpolated and hard-capped.
6. **Plateau smoothing** — smooths interior points of flat runs sitting at the outlier ceiling.
7. **Feature engineering** — calendar features (hour, month, day of week, season, weekend flag), cyclical sin/cos encodings, and lag (1h, 24h) + rolling (3h, 24h) pollutant features.
8. **AQI target construction** — computed from raw PM2.5/PM10 sub-index formulas (CPCB methodology), taking the max of the two sub-indices.
9. **Model training** — `RandomForestRegressor` (100 trees) on a chronological 80/20 split, evaluated with MAE and R².

Running the script also produces `feature_importance.png` (top 15 features) and `actual_vs_predicted.png` (scatter of predicted vs. actual AQI on the held-out set).

## Results

| Metric | Value |
|---|---|
| R² (held-out hours) | 0.983 |
| MAE | ~15.97 AQI points |
| Rows used | 18,968 (after gap-filling) |
| Features engineered | 49 |
| Train / test split | 15,155 / 3,789 hours |

The dominant feature is `pm2_5_lag1` (PM2.5 one hour prior) — unsurprising, since AQI is highly autocorrelated hour to hour, but the engineered calendar and rolling features meaningfully sharpen the forecast beyond a naive lag baseline.

## Running it locally

```bash
pip install pandas numpy scikit-learn matplotlib
python main_changes.py
```

This reads `delhi_aqi.csv`, runs the full pipeline, prints diagnostics for each cleaning step to the console, and writes `delhi_aqi_final.csv`, `delhi_aqi_feature_engineered.csv`, `feature_importance.png`, and `actual_vs_predicted.png`.

To view the dashboard, just open `index.html` in a browser — it reads its numbers from `data.js`, so no server is required.

## Data

Hourly pollutant data (CO, NO, NO₂, SO₂, O₃, PM2.5, PM10, NH₃) from Delhi monitoring stations, spanning November 2020 to January 2023 (~18.8K raw hourly rows before gap-filling).
