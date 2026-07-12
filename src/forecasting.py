from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_URL = (
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/235/"
    "individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip"
)

TARGET_OPTIONS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def ensure_zip_cached(url: str, cache_dir: str | Path = ".cache") -> Path:
    """Download the UCI source ZIP once and reuse it across app runs."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    zip_path = cache_path / "uci_household_power.zip"

    if not zip_path.exists():
        import requests

        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return zip_path


def _load_cached_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None

    if path.suffix == ".pkl":
        with open(path, "rb") as file:
            series = pickle.load(file)
    elif path.suffix == ".parquet":
        series = pd.read_parquet(path)["value"]
    else:
        return None

    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _save_series(series: pd.Series, path: Path) -> None:
    with open(path, "wb") as file:
        pickle.dump(series.sort_index(), file)


def load_aggregates_from_uci(
    url: str,
    target_col: str,
    cache_dir: str | Path = ".cache",
) -> tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    """
    Stream the UCI household power file and build compact daily/hourly aggregates.

    The raw dataset has more than two million rows, so this function reads it in
    chunks and caches only the analysis-ready series used by the dashboard.
    """
    if target_col not in TARGET_OPTIONS:
        raise ValueError(f"Unsupported target column: {target_col}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    daily_path = cache_path / f"daily_sum__{target_col}.pkl"
    hourly_path = cache_path / f"hourly_mean__{target_col}.pkl"
    legacy_daily_path = cache_path / f"daily_sum__{target_col}.parquet"
    legacy_hourly_path = cache_path / f"hourly_mean__{target_col}.parquet"

    daily_sum_full = _load_cached_series(daily_path)
    if daily_sum_full is None:
        daily_sum_full = _load_cached_series(legacy_daily_path)

    hourly_mean_full = _load_cached_series(hourly_path)
    if hourly_mean_full is None:
        hourly_mean_full = _load_cached_series(legacy_hourly_path)

    if daily_sum_full is not None and hourly_mean_full is not None:
        return (
            daily_sum_full,
            hourly_mean_full,
            daily_sum_full.index.min(),
            daily_sum_full.index.max(),
        )

    zip_path = ensure_zip_cached(url, cache_path)
    daily_sum_acc: dict[pd.Timestamp, float] = {}
    hourly_sum_acc: dict[pd.Timestamp, float] = {}
    hourly_cnt_acc: dict[pd.Timestamp, int] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_name = next((name for name in zf.namelist() if name.lower().endswith(".txt")), None)
        if txt_name is None:
            raise ValueError("ZIP archive did not contain a .txt file.")

        with zf.open(txt_name) as file:
            reader = pd.read_csv(
                file,
                sep=";",
                usecols=["Date", "Time", target_col],
                na_values="?",
                low_memory=False,
                chunksize=200_000,
            )

            for chunk in reader:
                dt = pd.to_datetime(
                    chunk["Date"] + " " + chunk["Time"],
                    dayfirst=True,
                    errors="coerce",
                )
                values = pd.to_numeric(chunk[target_col], errors="coerce")
                mask = dt.notna() & values.notna()
                if not mask.any():
                    continue

                dt = dt[mask]
                values = values[mask]

                day_sum = values.groupby(dt.dt.floor("D")).sum()
                for idx, val in day_sum.items():
                    daily_sum_acc[idx] = daily_sum_acc.get(idx, 0.0) + float(val)

                hour = dt.dt.floor("H")
                hour_sum = values.groupby(hour).sum()
                hour_count = values.groupby(hour).count()
                for idx, val in hour_sum.items():
                    hourly_sum_acc[idx] = hourly_sum_acc.get(idx, 0.0) + float(val)
                for idx, count in hour_count.items():
                    hourly_cnt_acc[idx] = hourly_cnt_acc.get(idx, 0) + int(count)

    daily_sum_full = pd.Series(daily_sum_acc).sort_index()
    hourly_sum_full = pd.Series(hourly_sum_acc).sort_index()
    hourly_cnt_full = pd.Series(hourly_cnt_acc).sort_index()
    hourly_mean_full = (hourly_sum_full / hourly_cnt_full).dropna()

    _save_series(daily_sum_full, daily_path)
    _save_series(hourly_mean_full, hourly_path)

    return (
        daily_sum_full,
        hourly_mean_full,
        daily_sum_full.index.min(),
        daily_sum_full.index.max(),
    )


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))
    denom = y_true.replace(0, np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def fit_sarima_model(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(float(train.iloc[-1]), index=test_index, name="Naive")


def seasonal_naive_forecast(
    history: pd.Series,
    test_index: pd.DatetimeIndex,
    season_length: int = 7,
) -> pd.Series:
    predictions = []
    values = history.copy()

    for timestamp in test_index:
        if len(values) >= season_length:
            prediction = float(values.iloc[-season_length])
        else:
            prediction = float(values.iloc[-1])
        predictions.append(prediction)
        values.loc[timestamp] = prediction

    return pd.Series(predictions, index=test_index, name="Seasonal naive")


def rolling_mean_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    window: int = 7,
) -> pd.Series:
    return pd.Series(float(train.tail(window).mean()), index=test_index, name=f"{window}-day mean")


def benchmark_forecasts(
    train: pd.Series,
    test: pd.Series,
    sarima_pred: pd.Series,
) -> pd.DataFrame:
    """Compare SARIMA against simple forecasting baselines."""
    forecasts = {
        "Naive": naive_forecast(train, test.index),
        "Seasonal naive": seasonal_naive_forecast(train, test.index),
        "7-day mean": rolling_mean_forecast(train, test.index),
        "SARIMA": sarima_pred,
    }

    rows = []
    for model_name, prediction in forecasts.items():
        metrics = evaluate_forecast(test, prediction)
        rows.append({"Model": model_name, **metrics})

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
