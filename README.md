# Energy Consumption Forecasting

Forecast household electricity consumption from historical meter readings using time-series aggregation, baseline model comparison, SARIMA forecasting, and an interactive Streamlit dashboard.

## Overview

This project uses the UCI Individual Household Electric Power Consumption dataset to explore domestic electricity usage patterns and forecast future daily consumption. The workflow is designed as a practical forecasting project rather than a single notebook experiment: raw readings are streamed from the source ZIP file, converted into reusable daily and hourly aggregates, evaluated against simple forecasting baselines, and presented in a dashboard for analysis and planning.

The dashboard supports multiple energy-related targets, including global active power, voltage, global intensity, and sub-metering variables.

## Business Problem

Household energy demand changes across hours, weekdays, weekends, and seasons. A forecasting workflow can help answer practical planning questions:

- What does typical daily consumption look like?
- Which days or periods show unusually high usage?
- Is demand trending upward or downward?
- Can a SARIMA model improve on simple history-based baselines?
- What should the next 7 to 180 days of demand look like?

## Dataset

Source: [UCI Machine Learning Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The raw dataset contains minute-level household electricity measurements from December 2006 to November 2010. Because the source file is large, it is not stored directly in this repository. The app downloads the ZIP file from UCI on first run and caches compact aggregate files under `.cache/`.

Key fields include:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

## Workflow

1. Stream the raw UCI text file in chunks.
2. Parse timestamp fields and selected energy target.
3. Build daily totals and hourly average profiles.
4. Cache aggregate series for fast repeated app runs.
5. Explore usage trends, peaks, weekday/weekend effects, and seasonality.
6. Split the daily series into train and holdout windows.
7. Compare SARIMA against simple baselines:
   - naive forecast
   - seasonal naive forecast
   - 7-day mean forecast
8. Forecast a user-selected future horizon.

## Forecasting Approach

The primary forecasting model is:

```text
SARIMA (1,1,1) x (1,1,1,7)
```

The weekly seasonal component reflects the household consumption pattern expected in daily data. The app does not assume SARIMA is automatically better; it benchmarks the model against simpler alternatives so performance is visible and easy to interpret.

Evaluation metrics:

- MAE: mean absolute error
- RMSE: root mean squared error
- MAPE: mean absolute percentage error

## Dashboard Features

- Select target energy metric
- Choose custom date range
- Review daily, hourly, weekday, weekend, and monthly usage patterns
- Identify highest-consumption days
- Generate descriptive usage insights
- Run SARIMA holdout forecasting
- Compare SARIMA with baseline forecasts
- Download future forecast results as CSV

## Project Structure

```text
Energy-Consumption-Forecasting/
|-- app.py
|-- src/
|   |-- __init__.py
|   `-- forecasting.py
|-- Notebook/
|   `-- Energy_Consumption_Forecasting.ipynb
|-- images/
|   |-- daily_consumption.png
|   |-- forecast_vs_actual.png
|   |-- global_active_power_over_time.png
|   `-- time_series_decomposition.png
|-- data/
|   `-- README.md
|-- requirements.txt
|-- .gitignore
|-- LICENSE
`-- README.md
```

## How To Run

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

On the first run, the app downloads the source ZIP file and builds cached aggregate files. Later runs are much faster because the dashboard reuses the cached daily and hourly series.

Run the test suite:

```bash
pytest
```

## Reproducibility Notes

- Raw data is downloaded from UCI rather than committed to the repository.
- Cached files are stored in `.cache/` and ignored by Git.
- The reusable forecasting and data-loading logic lives in `src/forecasting.py`.
- The Streamlit app is responsible for user interaction and visualization.

## Current Limitations

- SARIMA parameters are fixed rather than selected through automated search.
- Forecasting uses univariate time-series history only.
- Weather, holidays, occupancy, and price signals are not included.
- The current validation is a single holdout split, not full rolling-origin backtesting.

## Future Improvements

- Add rolling-origin cross-validation.
- Add lag-feature machine learning models such as Random Forest, XGBoost, or LightGBM.
- Add holiday and weather covariates.
- Save model benchmark outputs to `outputs/metrics.json`.
- Add broader tests for aggregate loading and Streamlit-facing transformations.
- Add dashboard screenshots after final visual QA.
