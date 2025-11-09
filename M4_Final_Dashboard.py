"""
M4_Main_Dashboard.py
AirAware - Final Streamlit Dashboard (combine M1, M2, M3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import os

# Try optional imports (statsmodels, joblib). Use fallbacks if unavailable.
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="AirAware - Main Dashboard", layout="wide")
st.title("🌍 AirAware — Main Dashboard")
st.markdown("Data Explorer • Forecast Engine • Alert System — combined")

# ---------------------------
# Helpers
# ---------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(name, model):
    """Save model using joblib if available; otherwise, skip with info."""
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if HAS_JOBLIB:
        try:
            joblib.dump(model, path)
        except Exception as e:
            st.warning(f"Could not save model {name}: {e}")
    else:
        st.info("joblib not available — models will not be persisted.")


def load_model(name):
    """Load model if present and joblib available."""
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if HAS_JOBLIB and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model {name}: {e}")
            return None
    return None


def categorize_aqi(aqi):
    try:
        aqi = float(aqi)
    except:
        return "Unknown", "gray"
    if aqi <= 50:
        return "Good", "green"
    if aqi <= 100:
        return "Moderate", "yellow"
    if aqi <= 200:
        return "Unhealthy for Sensitive", "orange"
    if aqi <= 300:
        return "Unhealthy", "red"
    if aqi <= 400:
        return "Very Unhealthy", "purple"
    return "Hazardous", "maroon"


def fallback_linear_forecast(series, steps):
    """Linear regression forecast using date ordinal -> simple fallback."""
    from sklearn.linear_model import LinearRegression
    s = series.dropna()
    if len(s) < 3:
        return np.repeat(s.iloc[-1] if len(s) else 0, steps)
    X = np.array([d.toordinal() for d in s.index]).reshape(-1, 1)
    y = s.values
    lr = LinearRegression().fit(X, y)
    last = s.index.max()
    future = [last + timedelta(days=i) for i in range(1, steps + 1)]
    Xf = np.array([d.toordinal() for d in future]).reshape(-1, 1)
    preds = lr.predict(Xf)
    return pd.Series(preds, index=future)


def fallback_moving_average(series, steps, window=7):
    val = series.dropna().rolling(window=window, min_periods=1).mean().iloc[-1]
    last = series.dropna().index.max() if len(series.dropna()) else pd.Timestamp.today()
    future_dates = [last + timedelta(days=i) for i in range(1, steps + 1)]
    return pd.Series(np.repeat(val, steps), index=future_dates)


# ---------------------------
# File upload / load default
# ---------------------------
st.sidebar.header("📁 Data & Controls")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (demo)")

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df = None
elif use_sample:
    # sample dataset
    rng = pd.date_range(start="2023-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "Date": rng,
        "City": np.random.choice(["Delhi", "Ahmedabad", "Mumbai"], size=len(rng)),
        "PM2.5": np.abs(np.random.normal(100, 40, size=len(rng))).round(2),
        "PM10": np.abs(np.random.normal(150, 60, size=len(rng))).round(2),
        "AQI": np.clip(np.random.normal(150, 50, len(rng)), 10, 400).round(1)
    })
    st.sidebar.success("Sample data loaded")
else:
    df = None
    st.sidebar.info("Upload or enable sample data to proceed.")

# ---------------------------
# Validate & normalize df
# ---------------------------
if df is not None:
    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # find date column (case-insensitive contains 'date')
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        st.error("No date column found (column name containing 'date'). Please upload a correct CSV.")
        st.stop()

    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Normalize duplicate-date handling:
    # If there are multiple rows per date, aggregate numeric columns by daily mean
    if df[date_col].duplicated().any():
        st.warning("Duplicate timestamps/dates detected. Aggregating duplicates by daily mean.")
        # convert to date-only (preserve time if unique times needed)
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        # groupby Date and mean for numeric columns; keep non-numeric by first
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_cols + [date_col]]
        grouped = df.groupby(date_col).agg({**{c: "mean" for c in numeric_cols}, **{c: "first" for c in non_numeric}})
        grouped = grouped.reset_index().rename(columns={date_col: "Date"})
        df = grouped
    else:
        df = df.rename(columns={date_col: "Date"})
        # ensure Date column normalized (no times)
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # find city/station column
    city_col = None
    if "City" in df.columns:
        city_col = "City"
    else:
        for c in df.columns:
            if any(k in c.lower() for k in ["station", "location", "city"]):
                city_col = c
                break

    # detect pollutants
    pollutant_candidates = [c for c in df.columns if any(p in c.upper() for p in ["PM2.5", "PM2_5", "PM25", "PM10", "O3", "NO2", "SO2", "CO", "AQI"])]
    # normalize PM2.5 variations
    if "PM2_5" in df.columns and "PM2.5" not in df.columns:
        df["PM2.5"] = df["PM2_5"]
    if "PM25" in df.columns and "PM2.5" not in df.columns:
        df["PM2.5"] = df["PM25"]

    has_aqi = "AQI" in df.columns

    # ---------------------------
    # Sidebar controls
    # ---------------------------
    st.sidebar.markdown("### Filters")
    if city_col:
        cities = ["All"] + sorted(df[city_col].dropna().unique().tolist())
    else:
        cities = ["All"]
    selected_city = st.sidebar.selectbox("Station / City", options=cities, index=0)

    # timeframe selection
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input("Time Range", [min_date, max_date])

    # pollutant selection
    default_pollutants = [p for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"] if p in df.columns]
    if default_pollutants:
        pollutant_list = default_pollutants
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        pollutant_list = [c for c in numeric_cols if c not in ["AQI"]][:4] if numeric_cols else []

    if not pollutant_list:
        st.error("No pollutant or numeric columns found to analyze.")
        st.stop()

    selected_pollutant = st.sidebar.selectbox("Pollutant", pollutant_list)
    forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 14, 7)
    refresh = st.sidebar.button("Update / Apply Filters")

    # Filter data
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if selected_city != "All" and city_col:
        df_filtered = df_filtered[df_filtered[city_col] == selected_city]

    st.write(f"### Data preview ({selected_city}) — {len(df_filtered)} records")
    st.dataframe(df_filtered.head(), use_container_width=True)

    # ---------------------------
    # Top layout: AQI Gauge + Quick stats
    # ---------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🌡️ Current AQI")
        if has_aqi and not df_filtered.empty:
            latest_aqi = df_filtered["AQI"].dropna().iloc[-1]
            prev_aqi_vals = df_filtered["AQI"].dropna()
            prev_ref = prev_aqi_vals.iloc[-2] if len(prev_aqi_vals) > 1 else latest_aqi
            status, color = categorize_aqi(latest_aqi)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(latest_aqi),
                delta={'reference': float(prev_ref)},
                title={'text': f"AQI: {status}"},
                gauge={'axis': {'range': [0, 500]},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgreen"},
                           {'range': [51, 100], 'color': "yellow"},
                           {'range': [101, 200], 'color': "orange"},
                           {'range': [201, 300], 'color': "red"},
                           {'range': [301, 500], 'color': "purple"}
                       ],
                       'bar': {'color': color}}
            ))
            st.plotly_chart(gauge, use_container_width=True)
            st.write(f"Latest AQI: **{latest_aqi:.1f}** — {status}")
        else:
            st.warning("AQI not available for selected filters.")

    with col2:
        st.subheader("📊 Quick Stats & Alerts")
        if not df_filtered.empty and selected_pollutant in df_filtered.columns:
            mean_val = df_filtered[selected_pollutant].mean()
            max_val = df_filtered[selected_pollutant].max()
            st.metric(label=f"Avg {selected_pollutant}", value=f"{mean_val:.2f}")
            st.metric(label=f"Max {selected_pollutant}", value=f"{max_val:.2f}")

            # Simple alerts based on thresholds (adjustable)
            alerts = []
            thresholds = {"PM2.5": 60, "PM10": 100, "O3": 120}
            if selected_pollutant in thresholds:
                if df_filtered[selected_pollutant].max() > thresholds[selected_pollutant]:
                    alerts.append(f"{selected_pollutant} exceeded threshold ({thresholds[selected_pollutant]}) in selected range.")
            if has_aqi and df_filtered["AQI"].max() > 200:
                alerts.append("AQI exceeded 200 (very poor/hazardous) in selected range.")
            if alerts:
                for a in alerts:
                    st.error(f"⚠️ {a}")
            else:
                st.success("✅ No active high-level alerts in the selected range.")
        else:
            st.info("No pollutant data available for quick stats.")

    st.markdown("---")

    # ---------------------------
    # Pollutant Trends
    # ---------------------------
    st.subheader("📈 Pollutant Trends")
    if not df_filtered.empty:
        trend_pollutants = [p for p in ["PM2.5", "PM10", "O3"] if p in df_filtered.columns]
        if trend_pollutants:
            fig_trend = go.Figure()
            for p in trend_pollutants:
                fig_trend.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered[p], mode="lines", name=p))
            # WHO limits
            who_limits = {"PM2.5": 25, "PM10": 50, "O3": 100}
            for p in trend_pollutants:
                limit = who_limits.get(p, None)
                if limit:
                    fig_trend.add_hline(y=limit, line_dash="dash", line_color="red",
                                        annotation_text=f"WHO {p} limit", annotation_position="top left")
            fig_trend.update_layout(title=f"Pollutant trends ({selected_city})", xaxis_title="Date", yaxis_title="Concentration")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No standard pollutant columns (PM2.5, PM10, O3) present to plot.")
    else:
        st.info("No data to show pollutant trends.")

    st.markdown("---")

    # ---------------------------
    # Forecast (ARIMA if available, else fallback)
    # ---------------------------
    st.subheader(f"🔮 Forecast — {selected_pollutant} (Actual vs Forecast)")

    if selected_pollutant in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[selected_pollutant]):
        try:
            # prepare series: ensure unique date index (group by Date if duplicates)
            ser = df_filtered[["Date", selected_pollutant]].copy()
            ser = ser.groupby("Date").mean(numeric_only=True).reset_index()
            ser = ser.set_index("Date")[selected_pollutant].asfreq("D")
            # fill small gaps by interpolation
            ser = ser.interpolate(method='time').ffill().bfill()

            if ser.dropna().shape[0] < 30:
                st.warning("Not enough data for reliable ARIMA training (need ~30+ days). Using fallback forecast.")
                # fallback forecast: moving average or linear regression
                future_series = fallback_linear_forecast(ser, forecast_horizon)
            else:
                model_name = f"arima_{selected_city}_{selected_pollutant}".replace(" ", "_") if selected_city != "All" else f"arima_{selected_pollutant}"
                model = None
                if HAS_ARIMA:
                    model = load_model(model_name)
                    if model is None:
                        with st.spinner("Training ARIMA model..."):
                            # simple ARIMA order; could be tuned
                            arima_model = ARIMA(ser, order=(2, 1, 2))
                            model = arima_model.fit()
                            save_model(model_name, model)
                if model is not None and HAS_ARIMA:
                    try:
                        forecast = model.forecast(steps=forecast_horizon)
                        last_date = ser.index.max()
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D")
                        future_series = pd.Series(forecast, index=future_dates)
                    except Exception as e:
                        st.warning(f"ARIMA forecasting failed, using fallback. ({e})")
                        future_series = fallback_linear_forecast(ser, forecast_horizon)
                else:
                    # no ARIMA available — use fallback
                    future_series = fallback_linear_forecast(ser, forecast_horizon)

            # Plot observed recent + forecast
            recent = ser[-90:]  # up to last 90 days
            figf = go.Figure()
            figf.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Actual"))
            figf.add_trace(go.Scatter(x=future_series.index, y=future_series.values, mode="lines+markers", name="Forecast"))
            figf.update_layout(title=f"{selected_pollutant} — Actual vs Forecast ({selected_city})", xaxis_title="Date", yaxis_title=selected_pollutant)
            st.plotly_chart(figf, use_container_width=True)

            # numeric table
            forecast_df = pd.DataFrame({"Date": future_series.index, "Forecast": np.round(future_series.values, 2)})
            st.write("Forecast (next days):")
            st.dataframe(forecast_df.set_index("Date"), use_container_width=True)
        except Exception as e:
            st.error(f"Forecasting error: {e}")
    else:
        st.info("Selected pollutant not numeric or not present — cannot forecast.")

    st.markdown("---")

    # ---------------------------
    # Admin Mode: Upload new data / retrain models
    # ---------------------------
    st.sidebar.header("⚙️ Admin")
    admin = st.sidebar.checkbox("Enable Admin Mode")
    if admin:
        st.sidebar.subheader("Upload new dataset")
        new_data = st.sidebar.file_uploader("Upload CSV to replace dataset (Admin)", type=["csv"], key="admin_upload")
        if new_data:
            try:
                new_df = pd.read_csv(new_data)
                st.sidebar.success("New dataset uploaded. Restart app to load it.")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {e}")

        st.sidebar.markdown("### Retrain models")
        if st.sidebar.button("Retrain ARIMA models for selected pollutant(s)"):
            retrain_pollutants = [p for p in ["PM2.5", "PM10", "O3"] if p in df_filtered.columns]
            with st.spinner("Retraining models..."):
                for p in retrain_pollutants:
                    try:
                        s = df_filtered.groupby("Date").mean(numeric_only=True).reset_index().set_index("Date")[p].asfreq("D")
                        s = s.interpolate(method='time').ffill().bfill()
                        if s.dropna().shape[0] < 30:
                            st.warning(f"Skipping retrain for {p} — not enough data.")
                            continue
                        if HAS_ARIMA:
                            m = ARIMA(s, order=(2, 1, 2)).fit()
                            save_model(f"arima_{selected_city}_{p}".replace(" ", "_"), m)
                        else:
                            st.info("statsmodels not available — cannot retrain ARIMA.")
                    except Exception as e:
                        st.error(f"Retrain error for {p}: {e}")
                st.success("Retrain complete (where possible). Models saved to /models.")
else:
    st.info("Upload dataset from the left sidebar or enable sample data to start.")
