import pandas as pd

from src.forecasting import benchmark_forecasts, evaluate_forecast


def test_evaluate_forecast_returns_expected_metrics():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    actual = pd.Series([10.0, 20.0, 30.0], index=index)
    predicted = pd.Series([12.0, 18.0, 33.0], index=index)

    metrics = evaluate_forecast(actual, predicted)

    assert set(metrics) == {"MAE", "RMSE", "MAPE"}
    assert round(metrics["MAE"], 2) == 2.33
    assert metrics["RMSE"] > metrics["MAE"]
    assert metrics["MAPE"] > 0


def test_benchmark_forecasts_includes_baselines_and_sarima():
    index = pd.date_range("2024-01-01", periods=21, freq="D")
    series = pd.Series(range(21), index=index, dtype=float)
    train = series.iloc[:14]
    test = series.iloc[14:]
    sarima_pred = test.copy()

    benchmark = benchmark_forecasts(train, test, sarima_pred)

    assert set(benchmark["Model"]) == {"Naive", "Seasonal naive", "7-day mean", "SARIMA"}
    assert benchmark.iloc[0]["Model"] == "SARIMA"
    assert benchmark.iloc[0]["RMSE"] == 0
