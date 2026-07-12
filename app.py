import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import (
    DEFAULT_DATA_URL,
    TARGET_OPTIONS as FORECAST_TARGET_OPTIONS,
    benchmark_forecasts,
    evaluate_forecast,
    fit_sarima_model,
    load_aggregates_from_uci,
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.68)"
BORDER = "rgba(15,23,42,0.08)"
ACCENT = "#2563EB"


# =========================================================
# STYLING
# =========================================================
def apply_styles():
    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
    color: {TEXT};
}}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
.block-container,
.block-container *,
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {{
    color: {TEXT};
}}

.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1360px;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

section[data-testid="stSidebar"] > div {{
    background: {BG};
    border-right: 1px solid {BORDER};
}}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
[data-testid="stExpander"],
[data-testid="stExpander"] *,
[data-testid="stAlert"],
[data-testid="stAlert"] * {{
    color: {TEXT};
}}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: {MUTED};
}}

div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
div[data-baseweb="slider"] *,
div[data-baseweb="tab-list"] *,
button[kind],
button[kind] * {{
    color: {TEXT};
}}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea,
input {{
    background: {CARD};
    color: {TEXT};
    border-color: {BORDER};
}}

div[data-baseweb="tab"] {{
    color: {MUTED};
}}

div[data-baseweb="tab"][aria-selected="true"] {{
    color: {TEXT};
}}

[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *,
[data-testid="stTable"],
[data-testid="stTable"] * {{
    color: {TEXT};
}}

.card {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 18px 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}}

.kpi-card {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 16px 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    min-height: 118px;
}}

.kpi-title {{
    color: {MUTED};
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}}

.kpi-value {{
    color: {TEXT};
    font-size: 30px;
    font-weight: 800;
    line-height: 1.05;
}}

.kpi-subtitle {{
    color: {MUTED};
    font-size: 12px;
    margin-top: 8px;
}}

.hero {{
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(255,255,255,0.94));
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 24px 24px 18px 24px;
    margin-bottom: 1rem;
}}

.hero-title {{
    color: {TEXT};
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.55rem;
}}

.hero-sub {{
    color: {MUTED};
    font-size: 1rem;
    line-height: 1.75;
    max-width: 980px;
}}

.section-label {{
    color: {ACCENT};
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}

.small {{
    color: {MUTED};
    font-size: 12px;
}}

.metric-chip {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(15,23,42,0.05);
    border: 1px solid rgba(15,23,42,0.06);
    font-size: 12px;
    font-weight: 700;
    color: {TEXT};
    margin-right: 8px;
    margin-bottom: 8px;
}}

.insight-box {{
    background: rgba(37,99,235,0.04);
    color: {TEXT};
    border: 1px solid rgba(37,99,235,0.10);
    border-radius: 16px;
    padding: 14px 16px;
}}
</style>
""",
        unsafe_allow_html=True,
    )


apply_styles()


# =========================================================
# UI HELPERS
# =========================================================
def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(label: str, title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        if not np.isfinite(float(x)):
            return "—"
    except Exception:
        return "—"

    x = float(x)
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:.{digits}f}"


def safe_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    try:
        if not np.isfinite(float(x)):
            return "—"
    except Exception:
        return "—"
    return f"{float(x):.{digits}f}%"


def badge_row(items: list[str]) -> None:
    html = "".join([f'<span class="metric-chip">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# DATA CONFIG
# =========================================================
DEFAULT_URL = DEFAULT_DATA_URL
TARGET_OPTIONS = FORECAST_TARGET_OPTIONS


@st.cache_data(show_spinner=False)
def load_aggregates(url: str, target_col: str):
    return load_aggregates_from_uci(url, target_col)


# =========================================================
# ANALYTICS HELPERS
# =========================================================
def eval_forecast(y_true: pd.Series, y_pred: pd.Series):
    metrics = evaluate_forecast(y_true, y_pred)
    return metrics["MAE"], metrics["RMSE"], metrics["MAPE"]


@st.cache_resource(show_spinner=False)
def fit_sarima_cached(
    train: pd.Series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
):
    return fit_sarima_model(train, order=order, seasonal_order=seasonal_order)


def build_profiles(daily_sum: pd.Series, hourly_mean: pd.Series):
    hour_profile = hourly_mean.groupby(hourly_mean.index.hour).mean()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_profile = (
        daily_sum.groupby(daily_sum.index.day_name())
        .mean()
        .reindex(weekday_order)
    )

    monthly_sum = daily_sum.resample("M").sum()

    top_days = daily_sum.sort_values(ascending=False).head(10).reset_index()
    top_days.columns = ["Date", "Total"]

    is_weekend = daily_sum.index.dayofweek >= 5
    weekday_avg = daily_sum[~is_weekend].mean()
    weekend_avg = daily_sum[is_weekend].mean()
    weekend_delta = (
        (weekend_avg / weekday_avg - 1) * 100
        if pd.notna(weekday_avg) and weekday_avg != 0
        else np.nan
    )

    return hour_profile, weekday_profile, monthly_sum, top_days, weekend_delta


def rolling_summary(daily_sum: pd.Series):
    roll7 = daily_sum.rolling(7, min_periods=1).mean()
    roll30 = daily_sum.rolling(30, min_periods=1).mean()
    return roll7, roll30


def compute_core_metrics(daily_sum: pd.Series, monthly_sum: pd.Series, weekend_delta: float):
    avg_daily = float(daily_sum.mean())
    peak_daily = float(daily_sum.max())
    low_daily = float(daily_sum.min())
    total_usage = float(daily_sum.sum())
    variability = (
        float((daily_sum.std(ddof=0) / daily_sum.mean()) * 100)
        if pd.notna(daily_sum.mean()) and daily_sum.mean() != 0
        else np.nan
    )
    peak_month = monthly_sum.idxmax().strftime("%B %Y") if len(monthly_sum) else "—"

    return {
        "days_in_view": int(len(daily_sum)),
        "avg_daily": avg_daily,
        "peak_daily": peak_daily,
        "low_daily": low_daily,
        "total_usage": total_usage,
        "variability_pct": variability,
        "weekend_delta": weekend_delta,
        "peak_month": peak_month,
    }


def build_actionable_insights(
    daily_sum: pd.Series,
    hour_profile: pd.Series,
    weekday_profile: pd.Series,
    monthly_sum: pd.Series,
    weekend_delta: float,
):
    insights = []

    if len(hour_profile):
        peak_hour = int(hour_profile.idxmax())
        insights.append(
            f"Peak usage typically occurs around {peak_hour}:00, making this the strongest hourly demand window in the selected range."
        )

    if weekday_profile.dropna().shape[0] > 0:
        best_day = weekday_profile.idxmax()
        low_day = weekday_profile.idxmin()
        insights.append(
            f"Highest average daily usage occurs on {best_day}, while the lowest average occurs on {low_day}, which helps identify routine-driven demand patterns."
        )

    if len(monthly_sum) > 0:
        peak_month = monthly_sum.idxmax().strftime("%B %Y")
        insights.append(
            f"The highest monthly total occurs in {peak_month}, pointing to the strongest seasonal concentration of consumption."
        )

    if np.isfinite(weekend_delta):
        direction = "higher" if weekend_delta > 0 else "lower"
        insights.append(
            f"Weekend usage is {abs(weekend_delta):.1f}% {direction} than weekday usage, suggesting a measurable behaviour shift between working days and weekends."
        )

    mean_val = float(daily_sum.mean())
    std_val = float(daily_sum.std(ddof=0))
    if np.isfinite(mean_val) and mean_val != 0 and np.isfinite(std_val):
        cv = (std_val / mean_val) * 100
        insights.append(
            f"Day-to-day variability is {cv:.1f}% of the average daily total, which gives a useful indication of how stable or volatile the series is."
        )

    if len(daily_sum) >= 60:
        first = float(daily_sum.iloc[:30].mean())
        last = float(daily_sum.iloc[-30:].mean())
        if np.isfinite(first) and first != 0 and np.isfinite(last):
            change = (last / first - 1) * 100
            direction = "increased" if change > 0 else "decreased"
            insights.append(
                f"Average daily usage has {direction} by {abs(change):.1f}% when comparing the first 30 days with the last 30 days in the selected window."
            )

    return insights


def prepare_distribution_df(daily_sum: pd.Series) -> pd.DataFrame:
    out = daily_sum.reset_index()
    out.columns = ["Date", "DailyTotal"]
    out["Month"] = out["Date"].dt.strftime("%Y-%m")
    out["Weekday"] = out["Date"].dt.day_name()
    return out


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Energy Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "Insights", "Forecast"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data source")
data_url = st.sidebar.text_input("UCI ZIP URL", value=DEFAULT_URL)

st.sidebar.divider()
st.sidebar.subheader("Target metric")
target_col = st.sidebar.selectbox("Column to analyse", options=TARGET_OPTIONS, index=0)

with st.spinner("Preparing aggregates. First run takes longer; later runs load faster from cache."):
    daily_sum_full, hourly_mean_full, min_dt, max_dt = load_aggregates(data_url, target_col)

default_start = max(min_dt, max_dt - pd.Timedelta(days=365))

st.sidebar.subheader("Date range")
date_range = st.sidebar.date_input(
    "Select range",
    value=(default_start.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start = min_dt
    end = max_dt

if start > end:
    st.error("Start date must be earlier than end date.")
    st.stop()

daily_sum = daily_sum_full.loc[(daily_sum_full.index >= start) & (daily_sum_full.index <= end)]
hourly_mean = hourly_mean_full.loc[(hourly_mean_full.index >= start) & (hourly_mean_full.index <= end)]

if daily_sum.empty or hourly_mean.empty:
    st.warning("No data is available for the selected range.")
    st.stop()

hour_profile, weekday_profile, monthly_sum, top_days, weekend_delta = build_profiles(daily_sum, hourly_mean)
roll7, roll30 = rolling_summary(daily_sum)
core_metrics = compute_core_metrics(daily_sum, monthly_sum, weekend_delta)
dist_df = prepare_distribution_df(daily_sum)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Household Energy Consumption Dashboard</div>
        <div class="hero-sub">
            An interactive dashboard for analysing household electricity usage patterns across time,
            identifying peak demand behaviour, and forecasting future daily consumption using SARIMA.
            The app prioritises interpretable usage insights first, then supports forward-looking planning with a forecast view.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DASHBOARD SUMMARY
# =========================================================
if page == "Dashboard Summary":
    section_header(
        "Overview",
        "Consumption snapshot",
        "This page leads with behavioural usage signals and trend context rather than only raw dataset counts.",
    )

    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        kpi_card(
            "Days in View",
            f"{core_metrics['days_in_view']:,}",
            f"{daily_sum.index.min().date()} to {daily_sum.index.max().date()}",
        )
    with k2:
        kpi_card(
            "Average Daily Total",
            fmt_num(core_metrics["avg_daily"]),
            "Typical daily usage over the selected period",
        )
    with k3:
        kpi_card(
            "Peak Daily Total",
            fmt_num(core_metrics["peak_daily"]),
            f"Highest day: {daily_sum.idxmax().date()}",
        )
    with k4:
        kpi_card(
            "Total Usage",
            fmt_num(core_metrics["total_usage"]),
            "Sum across the selected date range",
        )

    st.write("")

    k5, k6, k7, k8 = st.columns(4, gap="large")
    with k5:
        kpi_card(
            "Weekend vs Weekday",
            safe_pct(core_metrics["weekend_delta"]),
            "Relative change vs weekday average",
        )
    with k6:
        kpi_card(
            "Lowest Daily Total",
            fmt_num(core_metrics["low_daily"]),
            f"Lowest day: {daily_sum.idxmin().date()}",
        )
    with k7:
        kpi_card(
            "Variability",
            safe_pct(core_metrics["variability_pct"]),
            "Coefficient of variation across daily totals",
        )
    with k8:
        kpi_card(
            "Peak Month",
            core_metrics["peak_month"],
            "Month with the highest total usage",
        )

    st.write("")

    left, right = st.columns([1.45, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Daily trend with rolling context")
        st.caption("Daily totals are shown alongside 7-day and 30-day rolling averages to reveal short-term noise and broader shifts.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_sum.index, y=daily_sum.values, name="Daily total"))
        fig.add_trace(go.Scatter(x=roll7.index, y=roll7.values, name="7-day average"))
        fig.add_trace(go.Scatter(x=roll30.index, y=roll30.values, name="30-day average"))
        fig.update_layout(
            height=440,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h"),
            xaxis_title="Date",
            yaxis_title=target_col,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What stands out")
        peak_hour = int(hour_profile.idxmax()) if len(hour_profile) else None
        peak_weekday = weekday_profile.idxmax() if weekday_profile.dropna().shape[0] else "—"

        notes = [
            f"- Average daily usage is **{fmt_num(core_metrics['avg_daily'])}**, with a peak of **{fmt_num(core_metrics['peak_daily'])}**.",
            f"- The highest month total occurs in **{core_metrics['peak_month']}**.",
        ]
        if peak_hour is not None:
            notes.append(f"- Usage typically peaks around **{peak_hour}:00**.")
        if peak_weekday != "—":
            notes.append(f"- The strongest weekday pattern occurs on **{peak_weekday}**.")
        if np.isfinite(core_metrics["weekend_delta"]):
            direction = "higher" if core_metrics["weekend_delta"] > 0 else "lower"
            notes.append(
                f"- Weekend usage is **{abs(core_metrics['weekend_delta']):.1f}% {direction}** than weekday usage."
            )

        st.write("\n".join(notes))
        st.write("")
        badge_row(
            [
                f"Metric: {target_col}",
                f"Range: {daily_sum.index.min().date()} to {daily_sum.index.max().date()}",
                "Forecast model: SARIMA",
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hour-of-day pattern")
        st.caption("Average hourly values across the selected range.")

        hp = pd.DataFrame({"Hour": hour_profile.index, "Average": hour_profile.values})
        fig = px.bar(hp, x="Hour", y="Average")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if not hp.empty:
            st.info(f"Typical peak hour: {int(hp.loc[hp['Average'].idxmax(), 'Hour'])}:00")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Weekday pattern")
        st.caption("Average daily totals grouped by weekday.")

        wp = pd.DataFrame({"Day": weekday_profile.index, "AverageDaily": weekday_profile.values})
        fig = px.bar(wp, x="Day", y="AverageDaily")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if weekday_profile.dropna().shape[0] > 0:
            st.info(
                f"Highest average day: {weekday_profile.idxmax()}. Lowest average day: {weekday_profile.idxmin()}."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Monthly totals")
        st.caption("Monthly total usage for a seasonal view.")

        ms = monthly_sum.reset_index()
        ms.columns = ["Month", "MonthlyTotal"]
        fig = px.bar(ms, x="Month", y="MonthlyTotal")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if not ms.empty:
            st.info(
                f"Highest month total: {pd.to_datetime(ms.loc[ms['MonthlyTotal'].idxmax(), 'Month']).strftime('%B %Y')}"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Glossary"):
        st.markdown(
            """
- Daily total: Sum of the selected metric across a full day.  
- Hour-of-day pattern: Average usage for each hour from 0 to 23.  
- Weekday pattern: Average daily total grouped by weekday.  
- Monthly total: Sum of daily totals for each month.  
- Variability: Coefficient of variation, used as a simple measure of day-to-day stability.  
            """.strip()
        )


# =========================================================
# INSIGHTS
# =========================================================
elif page == "Insights":
    section_header(
        "Interpretation",
        "Usage insights",
        "Auto-generated descriptive insights based on the currently selected metric and date range.",
    )

    insights = build_actionable_insights(
        daily_sum=daily_sum,
        hour_profile=hour_profile,
        weekday_profile=weekday_profile,
        monthly_sum=monthly_sum,
        weekend_delta=weekend_delta,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Key insights")
    if not insights:
        st.write("No insights could be generated for the current selection.")
    else:
        for item in insights[:8]:
            st.write(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.3, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Where peaks happen")
        st.caption("These charts highlight when higher usage tends to occur across hours and weekdays.")

        hp = pd.DataFrame({"Hour": hour_profile.index, "AverageHourly": hour_profile.values})
        fig1 = px.line(hp, x="Hour", y="AverageHourly", markers=True)
        fig1.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig1, use_container_width=True)

        wp = pd.DataFrame({"Day": weekday_profile.index, "AverageDaily": weekday_profile.values})
        fig2 = px.bar(wp, x="Day", y="AverageDaily")
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distribution and context")
        st.caption("A simple view of how daily totals are distributed within the selected range.")

        fig_hist = px.histogram(dist_df, x="DailyTotal", nbins=40)
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        st.subheader("Interpretation notes")
        st.write(
            "- Peaks and dips describe behaviour patterns, not causes.\n"
            "- External drivers such as weather, tariffs, occupancy, and appliance usage can materially affect interpretation.\n"
            "- Use the Forecast page for forward-looking planning based on daily totals."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    bottom_left, bottom_right = st.columns([1.15, 1.0], gap="large")

    with bottom_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top usage days")
        st.caption("The highest daily totals in the selected range.")
        st.dataframe(top_days, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick numbers")
        q1, q2 = st.columns(2, gap="large")
        with q1:
            kpi_card("Average Daily Total", fmt_num(core_metrics["avg_daily"]), "Selected range")
        with q2:
            kpi_card("Peak Daily Total", fmt_num(core_metrics["peak_daily"]), f"On {daily_sum.idxmax().date()}")

        st.write("")
        st.markdown(
            """
<div class="insight-box">
These insights are descriptive and pattern-based. They are most useful for highlighting where deeper operational or behavioural investigation should start.
</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FORECAST
# =========================================================
else:
    section_header(
        "Prediction",
        "Forecast model comparison",
        "Compare simple baselines with SARIMA (1,1,1) x (1,1,1,7), then generate future daily forecasts.",
    )

    series_forecast = daily_sum.asfreq("D").ffill().bfill()

    ctrl_left, ctrl_right = st.columns([1.2, 1.0], gap="large")

    with ctrl_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        split_pct = st.slider("Train split (%)", 60, 90, 80)
        horizon = st.number_input("Future forecast days", min_value=7, max_value=180, value=30)
        st.markdown("</div>", unsafe_allow_html=True)

    with ctrl_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Model configuration")
        badge_row(
            [
                "SARIMA (1,1,1)",
                "Seasonal (1,1,1,7)",
                f"Horizon: {int(horizon)} days",
            ]
        )
        run = st.button("Run forecast", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        st.info("Click Run forecast to generate holdout and future forecasts.")
        st.stop()

    split_idx = int(len(series_forecast) * (split_pct / 100))
    train = series_forecast.iloc[:split_idx]
    test = series_forecast.iloc[split_idx:]

    if len(train) < 30 or len(test) < 7:
        st.error("The selected range is too short for a stable forecast evaluation. Choose a wider date range.")
        st.stop()

    with st.spinner("Fitting SARIMA model..."):
        fitted = fit_sarima_cached(
            train=train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
        )

    start_i = len(train)
    end_i = len(train) + len(test) - 1
    pred_obj = fitted.get_prediction(start=start_i, end=end_i, dynamic=False)

    pred = pred_obj.predicted_mean
    pred.index = test.index

    ci = pred_obj.conf_int()
    ci.index = test.index

    mae, rmse, mape = eval_forecast(test, pred)
    benchmark_df = benchmark_forecasts(train, test, pred)

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("MAE", fmt_num(mae), "Average absolute error")
    with m2:
        kpi_card("RMSE", fmt_num(rmse), "Penalises larger forecast errors")
    with m3:
        kpi_card("MAPE", safe_pct(mape, 2), "Average percentage error")
    with m4:
        kpi_card("Forecast Horizon", f"{int(horizon)} days", "Future projection window")

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model benchmark")
    st.caption(
        "SARIMA is evaluated against simple forecasting baselines so model value is visible, not assumed."
    )
    display_benchmark = benchmark_df.copy()
    for col in ["MAE", "RMSE", "MAPE"]:
        display_benchmark[col] = display_benchmark[col].map(lambda value: f"{value:,.2f}")
    st.dataframe(display_benchmark, use_container_width=True, hide_index=True)
    best_model = benchmark_df.iloc[0]["Model"]
    st.markdown(
        f"""
<div class="insight-box">
Best holdout model by RMSE: <strong>{best_model}</strong>. Use this table to check whether SARIMA adds value beyond simple demand-history baselines.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Holdout forecast vs actual")
    st.caption("Compares train history, actual holdout values, and SARIMA predictions with confidence intervals.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast"))

    try:
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        fig.add_trace(go.Scatter(x=ci.index, y=upper, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=ci.index, y=lower, mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))
    except Exception:
        pass

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
        xaxis_title="Date",
        yaxis_title=target_col,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"Future forecast: next {int(horizon)} days")

        fut_obj = fitted.get_forecast(steps=int(horizon))
        fut = fut_obj.predicted_mean
        fci = fut_obj.conf_int()

        future_df = pd.DataFrame(
            {
                "Date": fut.index,
                "Forecast": fut.values,
                "Lower": fci.iloc[:, 0].values if fci is not None else np.nan,
                "Upper": fci.iloc[:, 1].values if fci is not None else np.nan,
            }
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series_forecast.index, y=series_forecast.values, name="History"))
        fig2.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Forecast"], name="Forecast"))

        try:
            fig2.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Upper"], mode="lines", line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))
        except Exception:
            pass

        fig2.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h"),
            xaxis_title="Date",
            yaxis_title=target_col,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Download future forecast CSV",
            data=future_df.to_csv(index=False).encode("utf-8"),
            file_name="energy_future_forecast.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Forecast interpretation")
        st.write(
            f"""
The current SARIMA model is fitted on **{split_pct}%** of the selected daily series and evaluated on the remaining holdout window.

- **MAE** shows average absolute forecast error.
- **RMSE** gives extra weight to larger misses.
- **MAPE** provides a percentage-based view of forecast accuracy.
- Confidence intervals widen when uncertainty increases.

This forecast is useful for directional planning and scenario comparison rather than exact operational guarantees.
            """.strip()
        )

        st.divider()
        st.subheader("Forecast diagnostics")
        residuals = test - pred
        st.write(
            f"- Mean residual: **{fmt_num(residuals.mean())}**\n"
            f"- Residual std dev: **{fmt_num(residuals.std(ddof=0))}**\n"
            f"- Test observations: **{len(test):,}**"
        )
        st.markdown("</div>", unsafe_allow_html=True)
