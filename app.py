import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import requests
import zipfile
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Page config (LIGHT only)
# =========================
st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"


def apply_styles():
    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 2.6rem;   /* avoids title clipping */
    padding-bottom: 1.5rem;
}}
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.small {{
    color: {MUTED};
    font-size: 12px;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div style="color:{MUTED}; font-weight:600; font-size:14px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


apply_styles()

# =========================
# Data (fast approach)
# - Download ZIP once
# - Stream txt in chunks
# - Compute DAILY SUM + HOURLY MEAN without keeping 2M rows in memory
# - Cache aggregates to disk for future runs
# =========================
DEFAULT_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip"


def ensure_zip_cached(url: str) -> Path:
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    zip_path = cache_dir / "uci_household_power.zip"

    if not zip_path.exists():
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return zip_path


@st.cache_data(show_spinner=False)
def load_aggregates(url: str, target_col: str):
    """
    Returns:
      daily_sum_full (Series: daily totals)
      hourly_mean_full (Series: hourly mean values)
      min_dt, max_dt
    Uses disk cache to avoid recomputing.
    """
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    daily_path = cache_dir / f"daily_sum__{target_col}.parquet"
    hourly_path = cache_dir / f"hourly_mean__{target_col}.parquet"

    # Disk cache (fast path)
    if daily_path.exists() and hourly_path.exists():
        daily_sum_full = pd.read_parquet(daily_path)["value"]
        hourly_mean_full = pd.read_parquet(hourly_path)["value"]
        daily_sum_full.index = pd.to_datetime(daily_sum_full.index)
        hourly_mean_full.index = pd.to_datetime(hourly_mean_full.index)
        daily_sum_full = daily_sum_full.sort_index()
        hourly_mean_full = hourly_mean_full.sort_index()
        return daily_sum_full, hourly_mean_full, daily_sum_full.index.min(), daily_sum_full.index.max()

    # Compute (chunked)
    zip_path = ensure_zip_cached(url)

    daily_sum_acc = {}     # date -> sum
    hourly_sum_acc = {}    # hour -> sum
    hourly_cnt_acc = {}    # hour -> count

    with zipfile.ZipFile(zip_path, "r") as z:
        txt_name = next((n for n in z.namelist() if n.lower().endswith(".txt")), None)
        if txt_name is None:
            raise ValueError("ZIP did not contain a .txt file.")

        with z.open(txt_name) as f:
            reader = pd.read_csv(
                f,
                sep=";",
                usecols=["Date", "Time", target_col],
                na_values="?",
                low_memory=False,
                chunksize=200_000,
            )

            for chunk in reader:
                dt = pd.to_datetime(chunk["Date"] + " " + chunk["Time"], dayfirst=True, errors="coerce")
                v = pd.to_numeric(chunk[target_col], errors="coerce")

                mask = dt.notna() & v.notna()
                if not mask.any():
                    continue

                dt = dt[mask]
                v = v[mask]

                # Daily sum
                day = dt.dt.floor("D")
                day_sum = v.groupby(day).sum()
                for idx, val in day_sum.items():
                    daily_sum_acc[idx] = daily_sum_acc.get(idx, 0.0) + float(val)

                # Hourly mean via sum + count
                hour = dt.dt.floor("H")
                hour_sum = v.groupby(hour).sum()
                hour_cnt = v.groupby(hour).count()

                for idx, val in hour_sum.items():
                    hourly_sum_acc[idx] = hourly_sum_acc.get(idx, 0.0) + float(val)
                for idx, cnt in hour_cnt.items():
                    hourly_cnt_acc[idx] = hourly_cnt_acc.get(idx, 0) + int(cnt)

    daily_sum_full = pd.Series(daily_sum_acc).sort_index()
    hourly_sum_full = pd.Series(hourly_sum_acc).sort_index()
    hourly_cnt_full = pd.Series(hourly_cnt_acc).sort_index()

    hourly_mean_full = (hourly_sum_full / hourly_cnt_full).dropna()

    # Save to disk cache
    pd.DataFrame({"value": daily_sum_full}).to_parquet(daily_path)
    pd.DataFrame({"value": hourly_mean_full}).to_parquet(hourly_path)

    return daily_sum_full, hourly_mean_full, daily_sum_full.index.min(), daily_sum_full.index.max()


def eval_forecast(y_true: pd.Series, y_pred: pd.Series):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = y_true.replace(0, np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return mae, rmse, mape


def fit_sarima(train: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def safe_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.1f}%"


def build_profiles(daily_sum: pd.Series, hourly_mean: pd.Series):
    # Hour-of-day profile (average hourly value)
    hour_profile = hourly_mean.groupby(hourly_mean.index.hour).mean()

    # Weekday profile (average daily total)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_profile = daily_sum.groupby(daily_sum.index.day_name()).mean().reindex(weekday_order)

    # Monthly totals
    monthly_sum = daily_sum.resample("M").sum()

    # Top usage days
    top_days = daily_sum.sort_values(ascending=False).head(10).reset_index()
    top_days.columns = ["Date", "Total"]

    # Weekend vs weekday comparison
    is_weekend = daily_sum.index.dayofweek >= 5
    weekday_avg = daily_sum[~is_weekend].mean()
    weekend_avg = daily_sum[is_weekend].mean()
    weekend_delta = (weekend_avg / weekday_avg - 1) * 100 if weekday_avg != 0 else np.nan

    return hour_profile, weekday_profile, monthly_sum, top_days, weekend_delta


def build_actionable_insights(daily_sum: pd.Series, hour_profile: pd.Series, weekday_profile: pd.Series, monthly_sum: pd.Series, weekend_delta: float):
    insights = []

    # Peak hour
    peak_hour = int(hour_profile.idxmax()) if len(hour_profile) else None
    if peak_hour is not None:
        insights.append(f"Peak hour is typically around {peak_hour}:00. This is the highest average hourly usage window in the selected period.")

    # Best/worst weekday
    if weekday_profile.dropna().shape[0] > 0:
        best_day = weekday_profile.idxmax()
        low_day = weekday_profile.idxmin()
        insights.append(f"Highest average day is {best_day}, while the lowest average day is {low_day}. This helps prioritise when to target reduction strategies.")

    # Peak month
    if len(monthly_sum) > 0:
        peak_month = monthly_sum.idxmax().strftime("%B %Y")
        insights.append(f"Highest monthly total occurs in {peak_month}. This highlights the strongest seasonal concentration of consumption.")

    # Weekend delta
    if np.isfinite(weekend_delta):
        direction = "higher" if weekend_delta > 0 else "lower"
        insights.append(f"Weekend consumption is {abs(weekend_delta):.1f}% {direction} than weekdays. This indicates a behavioural shift in household usage patterns.")

    # Volatility
    mean_val = float(daily_sum.mean())
    std_val = float(daily_sum.std(ddof=0))
    if np.isfinite(mean_val) and mean_val != 0 and np.isfinite(std_val):
        cv = (std_val / mean_val) * 100
        insights.append(f"Day-to-day variability is about {cv:.1f}% of the average daily total. Higher variability usually means forecasting uncertainty is higher.")

    # Change over time (simple: first 30 vs last 30)
    if len(daily_sum) >= 60:
        first = float(daily_sum.iloc[:30].mean())
        last = float(daily_sum.iloc[-30:].mean())
        if np.isfinite(first) and first != 0 and np.isfinite(last):
            change = (last / first - 1) * 100
            trend_dir = "increased" if change > 0 else "decreased"
            insights.append(f"Average daily usage has {trend_dir} by {abs(change):.1f}% when comparing the first 30 days vs the last 30 days in the selected range.")

    return insights


# =========================
# Sidebar
# =========================
st.sidebar.title("Energy Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "Insights", "Forecast"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Data source")
data_url = st.sidebar.text_input("UCI ZIP URL", value=DEFAULT_URL)

st.sidebar.divider()
st.sidebar.subheader("Target")
target_col = st.sidebar.selectbox(
    "Column to analyse",
    options=[
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ],
    index=0,
)

with st.spinner("Preparing aggregates. First run takes longer. Later runs load quickly."):
    daily_sum_full, hourly_mean_full, min_dt, max_dt = load_aggregates(data_url, target_col)

# Default to last 12 months
default_start = (max_dt - pd.Timedelta(days=365)).date()

st.sidebar.subheader("Date range")
date_range = st.sidebar.date_input("Select range", value=(default_start, max_dt.date()))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start, end = min_dt, max_dt

# Filter series
daily_sum = daily_sum_full.loc[(daily_sum_full.index >= start) & (daily_sum_full.index <= end)]
hourly_mean = hourly_mean_full.loc[(hourly_mean_full.index >= start) & (hourly_mean_full.index <= end)]

if daily_sum.empty or hourly_mean.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# Derived profiles
hour_profile, weekday_profile, monthly_sum, top_days, weekend_delta = build_profiles(daily_sum, hourly_mean)

# =========================
# Header
# =========================
st.markdown("## Household Energy Consumption")
st.caption(
    "This dashboard starts with clear usage patterns (hours, weekdays, months) and then provides a SARIMA forecast using daily totals."
)
st.write("")


# =========================
# Dashboard Summary
# =========================
if page == "Dashboard Summary":
    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        kpi_card("Days in view", f"{len(daily_sum):,}", f"{daily_sum.index.min().date()} to {daily_sum.index.max().date()}")
    with k2:
        kpi_card("Average daily total", f"{daily_sum.mean():.2f}", "Typical day (sum of values)")
    with k3:
        kpi_card("Peak day total", f"{daily_sum.max():.2f}", f"Highest on {daily_sum.idxmax().date()}")
    with k4:
        kpi_card("Weekend vs weekday", safe_pct(weekend_delta), "Change relative to weekdays")

    st.write("")

    left, right = st.columns([1.6, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Daily trend")
        st.caption("Daily totals across the selected period. Useful for spotting spikes, dips, and longer-term shifts.")

        df_line = daily_sum.reset_index()
        df_line.columns = ["Date", "DailyTotal"]
        fig = px.line(df_line, x="Date", y="DailyTotal")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("High-usage days")
        st.caption("The highest daily totals in the selected period.")
        st.dataframe(top_days, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    a, b, c = st.columns([1.2, 1.2, 1.2], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hour-of-day pattern")
        st.caption("Average hourly usage by hour. Highlights typical peak times.")

        hp = pd.DataFrame({"Hour": hour_profile.index, "AvgHourly": hour_profile.values})
        fig = px.bar(hp, x="Hour", y="AvgHourly")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        peak_hour = int(hp.loc[hp["AvgHourly"].idxmax(), "Hour"])
        st.info(f"Typical peak hour: {peak_hour}:00")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Weekday pattern")
        st.caption("Average daily totals by weekday. Helps identify routine-driven consumption.")

        wp = pd.DataFrame({"Day": weekday_profile.index, "AvgDaily": weekday_profile.values})
        fig = px.bar(wp, x="Day", y="AvgDaily")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        best_day = wp.loc[wp["AvgDaily"].idxmax(), "Day"]
        low_day = wp.loc[wp["AvgDaily"].idxmin(), "Day"]
        st.info(f"Highest average day: {best_day}. Lowest average day: {low_day}.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Monthly totals")
        st.caption("Monthly total consumption. Useful for a seasonal overview.")

        ms = monthly_sum.reset_index()
        ms.columns = ["Month", "MonthlyTotal"]
        fig = px.bar(ms, x="Month", y="MonthlyTotal")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if not ms.empty:
            peak_month = pd.to_datetime(ms.loc[ms["MonthlyTotal"].idxmax(), "Month"]).strftime("%B %Y")
            st.info(f"Highest month total: {peak_month}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Glossary"):
        st.markdown(
            """
- Daily total: Sum of the selected metric across a full day.
- Hour-of-day pattern: Average usage for each clock hour (0–23).
- Weekday pattern: Average daily totals grouped by weekday.
- Monthly totals: Sum of daily totals for each month.
            """.strip()
        )


# =========================
# Insights (actionable, auto-generated)
# =========================
elif page == "Insights":
    st.markdown("### Insights")
    st.caption("Auto-generated insights based on the selected metric and date range. These are descriptive patterns observed in the data.")

    insights = build_actionable_insights(daily_sum, hour_profile, weekday_profile, monthly_sum, weekend_delta)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Actionable insights")
    if len(insights) == 0:
        st.write("No insights could be generated for the current selection.")
    else:
        for item in insights[:8]:
            st.write(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Where the peaks happen")
        st.caption("This view combines the hour and weekday profiles to support scheduling decisions.")

        hp = pd.DataFrame({"Hour": hour_profile.index, "AvgHourly": hour_profile.values})
        wp = pd.DataFrame({"Day": weekday_profile.index, "AvgDaily": weekday_profile.values})

        fig1 = px.line(hp, x="Hour", y="AvgHourly", markers=True, labels={"AvgHourly": "Average hourly usage"})
        fig1.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(wp, x="Day", y="AvgDaily", labels={"AvgDaily": "Average daily total"})
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key numbers")
        k1, k2 = st.columns(2, gap="large")
        with k1:
            kpi_card("Average daily total", f"{daily_sum.mean():.2f}", "Selected range")
        with k2:
            kpi_card("Peak daily total", f"{daily_sum.max():.2f}", f"On {daily_sum.idxmax().date()}")

        st.divider()
        st.subheader("Interpretation notes")
        st.write(
            "- Peaks and dips describe behaviour patterns in the selected period.\n"
            "- Recommendations should be validated with context (weather, occupancy, tariffs, household routines).\n"
            "- Use the Forecast page to estimate the next period based on daily totals."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Forecast (SARIMA; stable indexing)
# =========================
else:
    st.markdown("### Forecast")
    st.caption("Forecast is generated from daily totals using SARIMA (1,1,1) × (1,1,1,7).")

    # Force daily frequency for stable SARIMA indexing
    series_forecast = daily_sum.asfreq("D").ffill().bfill()

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        split_pct = st.slider("Train split (%)", 60, 90, 80)
        horizon = st.number_input("Future forecast days", min_value=7, max_value=180, value=30)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Model: SARIMA (1,1,1) × (1,1,1,7)")
        run = st.button("Run forecast", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        st.info("Click Run forecast to generate test and future forecasts.")
        st.stop()

    split_idx = int(len(series_forecast) * (split_pct / 100))
    train = series_forecast.iloc[:split_idx]
    test = series_forecast.iloc[split_idx:]

    with st.spinner("Fitting SARIMA model..."):
        fitted = fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

    # Predict using integer positions to avoid datetime label mismatch
    start_i = len(train)
    end_i = len(train) + len(test) - 1
    pred_obj = fitted.get_prediction(start=start_i, end=end_i, dynamic=False)

    pred = pred_obj.predicted_mean
    pred.index = test.index

    ci = pred_obj.conf_int()
    ci.index = test.index

    mae, rmse, mape = eval_forecast(test, pred)

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("MAE", f"{mae:.2f}", "Average absolute error")
    with m2:
        kpi_card("RMSE", f"{rmse:.2f}", "Penalises larger errors")
    with m3:
        kpi_card("MAPE", f"{mape:.2f}%", "Average percentage error")
    with m4:
        kpi_card("Horizon", f"{int(horizon)} days", "Future window")

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Test period: forecast vs actual")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (test)"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast"))

    # Confidence interval shading
    try:
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        fig.add_trace(go.Scatter(x=ci.index, y=upper, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=ci.index, y=lower, mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))
    except Exception:
        pass

    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"Future forecast: next {int(horizon)} days")

    fut_obj = fitted.get_forecast(steps=int(horizon))
    fut = fut_obj.predicted_mean
    fci = fut_obj.conf_int()

    fdf = pd.DataFrame({
        "Date": fut.index,
        "Forecast": fut.values,
        "Lower": fci.iloc[:, 0].values if fci is not None else np.nan,
        "Upper": fci.iloc[:, 1].values if fci is not None else np.nan,
    })

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=series_forecast.index, y=series_forecast.values, name="History"))
    fig2.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast"], name="Forecast"))

    try:
        fig2.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))
    except Exception:
        pass

    fig2.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download future forecast CSV",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="energy_future_forecast.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)
