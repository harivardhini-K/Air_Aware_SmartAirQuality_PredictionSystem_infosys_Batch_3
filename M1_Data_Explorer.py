# airaware_dashboard_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import timedelta

# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="AirAware - Air Quality Data Explorer", layout="wide")
st.title("🌍 AirAware - Air Quality Data Explorer (Enhanced)")
st.markdown("Use filters to explore air quality trends, correlations, distributions and insights.")

# -------------------------
# Sidebar controls & theme
# -------------------------
st.sidebar.header("🔧 Controls")

theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background: #0b0f16; color: #e6eef8; }
        .stButton>button { background-color: #1f2937; color: #e6eef8; border-radius: 8px; }
        .css-1v3fvcr { color: #e6eef8; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; color: #0b0f16; }
        .stButton>button { background-color: #f0f2f6; color: #0b0f16; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.sidebar.file_uploader("📂 Upload your Air Quality CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data (if no file)", value=False)

# -------------------------
# Load data helper
# -------------------------
@st.cache_data
def load_csv(uploaded):
    return pd.read_csv(uploaded)

@st.cache_data
def load_sample():
    # small synthetic/sample dataset to allow UI demo without file
    rng = pd.date_range("2023-01-01", periods=200, freq="D")
    data = pd.DataFrame({
        "date": rng,
        "City": np.random.choice(["CityA", "CityB", "CityC"], size=len(rng)),
        "PM2.5": np.abs(np.random.normal(60, 30, size=len(rng))).round(1),
        "PM10": np.abs(np.random.normal(90, 40, size=len(rng))).round(1),
        "NO2": np.abs(np.random.normal(30, 15, size=len(rng))).round(1),
        "AQI": np.clip(np.random.normal(120, 40, size=len(rng)), 10, 400).round(1),
        "Latitude": np.random.uniform(12.8, 13.2, size=len(rng)),
        "Longitude": np.random.uniform(77.4, 77.8, size=len(rng)),
    })
    return data

if uploaded_file:
    try:
        df = load_csv(uploaded_file)
        st.sidebar.success("File loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load CSV: {e}")
        df = pd.DataFrame()
elif use_sample:
    df = load_sample()
    st.sidebar.success("Loaded sample dataset")
else:
    df = pd.DataFrame()

if df.empty:
    st.info("👆 Upload a CSV or check 'Use sample data' to see the enhanced dashboard features.")
    st.stop()

# -------------------------
# Preprocess & column detection
# -------------------------
df = df.copy()
# Try to find a date column
date_col = None
for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        date_col = c
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Location column
location_col = None
for candidate in ["City", "city", "Location", "location", "Station", "station"]:
    if candidate in df.columns:
        location_col = candidate
        break

# Pollutant columns heuristic
pollutant_columns = [c for c in df.columns if any(p in c.upper() for p in ["PM", "NO", "SO", "CO", "O3", "AQI", "SO2", "NO2"])]
# Fallback: numeric columns except lat/lon/date
if not pollutant_columns:
    pollutant_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    pollutant_columns = [c for c in pollutant_columns if c.lower() not in ("latitude", "longitude")]

# lat/lon detection
has_latlon = {"lat": None, "lon": None}
for lat_name in ["Latitude", "latitude", "Lat", "lat"]:
    for lon_name in ["Longitude", "longitude", "Lon", "lon"]:
        if lat_name in df.columns and lon_name in df.columns:
            has_latlon["lat"], has_latlon["lon"] = lat_name, lon_name
            break
    if has_latlon["lat"]:
        break

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.markdown("### Filters")
selected_location = "All"
if location_col:
    locations = sorted(df[location_col].dropna().unique().tolist())
    selected_location = st.sidebar.selectbox("Select Location", ["All"] + locations)

selected_pollutants = st.sidebar.multiselect("Select Pollutants (for charts)", pollutant_columns, default=pollutant_columns[:3] if pollutant_columns else [])

# Date range
if date_col:
    if df[date_col].notna().any():
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
    else:
        min_date = pd.to_datetime("2020-01-01").date()
        max_date = pd.to_datetime("2020-12-31").date()

    start_date, end_date = st.sidebar.date_input("📆 Date range", value=[min_date, max_date])
else:
    start_date, end_date = None, None

# Compare cities option
enable_compare = st.sidebar.checkbox("Compare Two Locations / Cities", value=False)
compare_locations = []
if enable_compare and location_col:
    compare_locations = st.sidebar.multiselect("Select up to 2 locations to compare", locations, default=locations[:2])
    if len(compare_locations) > 2:
        st.sidebar.warning("Pick at most 2 locations for side-by-side comparison.")
# Apply filters button
apply_filters = st.sidebar.button("✅ Apply Filters")

# -------------------------
# Filtering the dataframe
# -------------------------
if apply_filters:
    filtered_df = df.copy()
    if selected_location != "All" and location_col:
        filtered_df = filtered_df[filtered_df[location_col] == selected_location]
    if date_col and start_date and end_date:
        filtered_df = filtered_df[(filtered_df[date_col] >= pd.to_datetime(start_date)) & (filtered_df[date_col] <= pd.to_datetime(end_date))]
else:
    # default view: not filtered
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No data available after applying filters. Try removing filters or choose a broader date range.")
    st.stop()

# -------------------------
# Summary metrics & top row
# -------------------------
st.subheader("📈 Key Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

# Average AQI
with col1:
    if "AQI" in filtered_df.columns:
        avg_aqi = filtered_df["AQI"].dropna().mean()
        st.metric("Average AQI", f"{avg_aqi:.1f}")
    else:
        st.metric("Average AQI", "N/A")

# Highest pollutant by mean
with col2:
    if selected_pollutants:
        numeric_present = [p for p in selected_pollutants if p in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[p])]
        if numeric_present:
            highest = filtered_df[numeric_present].mean().idxmax()
            highest_val = filtered_df[numeric_present].mean().max()
            st.metric("Highest (by mean)", f"{highest}: {highest_val:.1f}")
        else:
            st.metric("Highest (by mean)", "N/A")
    else:
        st.metric("Highest (by mean)", "N/A")

# Data records
with col3:
    st.metric("Data Records", f"{len(filtered_df):,}")

# Unique locations in filtered data
with col4:
    if location_col:
        st.metric("Locations", f"{filtered_df[location_col].nunique():,}")
    else:
        st.metric("Locations", "N/A")

# -------------------------
# Tabs: Data / Charts / Insights / Compare
# -------------------------
tabs = st.tabs(["📊 Data View", "📈 Charts", "🧠 Insights", "🧩 Tools & Export"])
tab_data, tab_charts, tab_insights, tab_tools = tabs

# -------------------------
# Tab: Data View
# -------------------------
with tab_data:
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.reset_index(drop=True).head(250))

    st.markdown("### Data Quality Check")
    missing_data = filtered_df.isnull().sum()
    completeness_percentage = 100 - (missing_data.sum() / (filtered_df.shape[0] * filtered_df.shape[1]) * 100)
    st.write("Missing values per column:")
    st.table(missing_data.reset_index().rename(columns={"index": "column", 0: "missing"}))
    st.progress(int(np.clip(completeness_percentage, 0, 100)))
    st.write(f"✅ Data Completeness: **{completeness_percentage:.2f}%**")

# -------------------------
# Tab: Charts
# -------------------------
with tab_charts:
    st.subheader("Charts & Analysis")

    # Time series for selected pollutants
    if date_col and selected_pollutants:
        st.markdown("#### Time Series")
        for pol in selected_pollutants:
            if pol in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pol]):
                fig = px.line(filtered_df.sort_values(by=date_col), x=date_col, y=pol, title=f"Time Series — {pol}", labels={date_col: "Date"})
                st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap using Plotly (works without seaborn)
    st.markdown("#### Pollutant Correlations")
    corr_cols = [c for c in selected_pollutants if c in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[c])]
    if len(corr_cols) >= 2:
        corr = filtered_df[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least two numeric pollutant columns to view correlation matrix.")

    # Distribution histograms
    st.markdown("#### Distribution Analysis")
    for pol in selected_pollutants:
        if pol in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pol]):
            fig = px.histogram(filtered_df, x=pol, nbins=30, title=f"Distribution — {pol}")
            st.plotly_chart(fig, use_container_width=True)

    # Map if lat/lon available
    if has_latlon["lat"]:
        st.markdown("#### Geographic Map")
        try:
            map_col = "AQI" if "AQI" in filtered_df.columns else (selected_pollutants[0] if selected_pollutants else None)
            if map_col is None:
                st.info("No pollutant/AQI found to color the map.")
            else:
                mfig = px.scatter_mapbox(
                    filtered_df.dropna(subset=[has_latlon["lat"], has_latlon["lon"]]),
                    lat=has_latlon["lat"],
                    lon=has_latlon["lon"],
                    color=map_col,
                    hover_name=location_col if location_col else None,
                    size=map_col if pd.api.types.is_numeric_dtype(filtered_df[map_col]) else None,
                    zoom=6,
                    height=500,
                    title="Air Quality Map"
                )
                mfig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(mfig, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {e}")
    else:
        st.info("Latitude/Longitude columns not found. Add 'Latitude' and 'Longitude' columns to enable map.")

    # Anomaly detection (z-score)
    st.markdown("#### Anomaly Detection (z-score)")
    for pol in selected_pollutants:
        if pol in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pol]):
            series = filtered_df[pol].dropna()
            if len(series) >= 10:
                zscores = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
                anomalies = series[np.abs(zscores) > 2.5]
                st.write(f"**{pol}** — anomalies found: {len(anomalies)}")
                if not anomalies.empty:
                    st.dataframe(pd.DataFrame({pol: anomalies}).head(10))
            else:
                st.write(f"**{pol}** — not enough data for anomaly detection (need >=10)")

    # Forecasting (simple moving average or linear trend)
    st.markdown("#### Forecasting (simple)")
    forecast_target = None
    if "AQI" in filtered_df.columns:
        forecast_target = "AQI"
    elif selected_pollutants:
        # pick first numeric pollutant
        for p in selected_pollutants:
            if p in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[p]):
                forecast_target = p
                break

    if date_col and forecast_target:
        ts = filtered_df[[date_col, forecast_target]].dropna().sort_values(by=date_col)
        ts = ts.groupby(date_col)[forecast_target].mean().reset_index()
        ts = ts.set_index(date_col).asfreq('D').interpolate()  # daily frequency & interpolate missing
        if len(ts) >= 10:
            window = min(7, max(3, len(ts)//10))
            ts['MA'] = ts[forecast_target].rolling(window=window, min_periods=1).mean()
            # linear trend projection for next 7 days using polyfit
            y = ts[forecast_target].values
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, deg=1)
            future_x = np.arange(len(y), len(y) + 7)
            future_y = np.polyval(coeffs, future_x)
            future_dates = [ts.index[-1] + timedelta(days=i) for i in range(1, 8)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts[forecast_target], name="Observed"))
            fig.add_trace(go.Scatter(x=ts.index, y=ts['MA'], name=f"{window}-day MA"))
            fig.add_trace(go.Scatter(x=future_dates, y=future_y, name="Linear Forecast (7 days)", line=dict(dash='dash')))
            fig.update_layout(title=f"Forecast for {forecast_target}", xaxis_title="Date", yaxis_title=forecast_target)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough daily data for forecasting (need >=10 days)")
    else:
        st.info("No numeric pollutant/AQI column available for forecasting.")

# -------------------------
# Tab: Insights
# -------------------------
with tab_insights:
    st.subheader("🧠 Smart Insights")
    insights = []

    # Average AQI interpretation
    if "AQI" in filtered_df.columns:
        avg_aqi = filtered_df["AQI"].dropna().mean()
        if avg_aqi < 50:
            category = "Good"
        elif avg_aqi < 100:
            category = "Moderate"
        elif avg_aqi < 200:
            category = "Poor"
        elif avg_aqi < 300:
            category = "Very Poor"
        else:
            category = "Hazardous"
        insights.append(f"Average AQI is {avg_aqi:.1f}, which falls in **{category}** category.")

        # worst day
        try:
            worst_idx = filtered_df["AQI"].idxmax()
            worst_day = filtered_df.loc[worst_idx, date_col] if date_col else None
            worst_loc = filtered_df.loc[worst_idx, location_col] if location_col else None
            if worst_day is not None:
                insights.append(f"Worst recorded AQI in the filtered range was **{filtered_df.loc[worst_idx,'AQI']:.1f}** on **{pd.to_datetime(worst_day).date()}** at **{worst_loc}**.")
        except Exception:
            pass

    # Most variable pollutant
    numeric_pollutants = [c for c in selected_pollutants if c in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[c])]
    if numeric_pollutants:
        var_poll = filtered_df[numeric_pollutants].std().idxmax()
        insights.append(f"Most variable pollutant (by standard deviation) is **{var_poll}**.")
    else:
        insights.append("No numeric pollutant selected to compute variability.")

    # If comparing two cities, short comparison summary
    if enable_compare and location_col and len(compare_locations) == 2:
        loc1, loc2 = compare_locations
        d1 = filtered_df[filtered_df[location_col] == loc1]
        d2 = filtered_df[filtered_df[location_col] == loc2]
        if not d1.empty and not d2.empty:
            mean1 = d1["AQI"].mean() if "AQI" in d1.columns else np.nan
            mean2 = d2["AQI"].mean() if "AQI" in d2.columns else np.nan
            insights.append(f"Comparison: **{loc1}** avg AQI = {mean1:.1f} vs **{loc2}** avg AQI = {mean2:.1f}.")
        else:
            insights.append("Comparison could not be performed due to missing data for selected locations.")

    # Anomaly summary (count across pollutants)
    anomaly_summary = {}
    for pol in numeric_pollutants:
        s = filtered_df[pol].dropna()
        if len(s) >= 10:
            z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
            anomaly_summary[pol] = int((np.abs(z) > 2.5).sum())
    if anomaly_summary:
        for k, v in anomaly_summary.items():
            insights.append(f"Anomalies detected in **{k}**: {v} points (z-score > 2.5).")

    # Show insights
    for i, text in enumerate(insights):
        st.write(f"{i+1}. {text}")

# -------------------------
# Tab: Tools & Export
# -------------------------
with tab_tools:
    st.subheader("Tools & Export")
    # Download filtered dataframe
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv, file_name="filtered_air_quality.csv", mime="text/csv")

    # AQI category coloring helper
    st.markdown("#### AQI Category Color Coding (if AQI present)")
    if "AQI" in filtered_df.columns:
        def aqi_category(aqi):
            try:
                a = float(aqi)
            except Exception:
                return "Unknown"
            if a <= 50: return "Good"
            if a <= 100: return "Moderate"
            if a <= 200: return "Poor"
            if a <= 300: return "Very Poor"
            return "Hazardous"

        categorized = filtered_df.copy()
        categorized["_AQI_Category"] = categorized["AQI"].apply(aqi_category)
        st.dataframe(categorized[[c for c in [date_col, location_col, "AQI", "_AQI_Category"] if c in categorized.columns]].head(200))
    else:
        st.info("AQI column not found. Add an 'AQI' column to use AQI category helper.")

    # Compare two locations visually
    if enable_compare and location_col and len(compare_locations) == 2:
        st.markdown("#### Side-by-side comparison chart")
        loc1, loc2 = compare_locations
        comp1 = filtered_df[filtered_df[location_col] == loc1]
        comp2 = filtered_df[filtered_df[location_col] == loc2]
        if date_col and not comp1.empty and not comp2.empty:
            # choose pollutant to compare
            compare_pollutant = st.selectbox("Select pollutant to compare", numeric_pollutants or list(filtered_df.select_dtypes(include=[np.number]).columns), index=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comp1[date_col], y=comp1[compare_pollutant], mode="lines+markers", name=loc1))
            fig.add_trace(go.Scatter(x=comp2[date_col], y=comp2[compare_pollutant], mode="lines+markers", name=loc2))
            fig.update_layout(title=f"Comparison: {loc1} vs {loc2} — {compare_pollutant}", xaxis_title="Date", yaxis_title=compare_pollutant)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for one or both selected locations to compare.")

    # Quick data stats download (summary)
    summary = filtered_df.describe(include='all').to_csv().encode("utf-8")
    st.download_button("⬇️ Download Data Summary (CSV)", data=summary, file_name="data_summary.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("Built with ❤️ — AirAware. Keep your data clean and sensors calibrated for best results.")
