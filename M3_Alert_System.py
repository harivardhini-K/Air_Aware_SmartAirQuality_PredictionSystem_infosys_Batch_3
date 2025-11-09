# ============================================================
# 🚨 AirAware - Air Quality Alert System (Enhanced Dashboard)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(page_title="AirAware - Alert System", layout="wide", page_icon="🚨")

st.title("🚨 AirAware - Air Quality Alert System")
st.markdown("Analyze current air quality, compare pollutants vs WHO limits, forecast short-term trends, and issue alerts.")

# ============================================================
# Sidebar Section
# ============================================================
st.sidebar.header("📂 Data & Filters")

uploaded_file = st.sidebar.file_uploader("Upload Air Quality Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # Detect Date Column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break

    # Detect City/Station Column
    location_col = None
    for col in df.columns:
        if 'city' in col.lower() or 'station' in col.lower():
            location_col = col
            break

    # Sidebar Filter for City
    if location_col:
        selected_location = st.sidebar.selectbox("🏙️ Select Location/Station", df[location_col].unique())
        df = df[df[location_col] == selected_location]

    # Sidebar Filter for Date Range
    if date_col:
        min_date, max_date = df[date_col].min(), df[date_col].max()
        start_date, end_date = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date])
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    # ============================================================
    # Tabs Layout
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💨 Pollutant Trends", "📅 Forecast", "⚠️ Alerts"])

    # ============================================================
    # TAB 1: Overview
    # ============================================================
    with tab1:
        st.subheader("📊 Current Air Quality Overview")

        if 'AQI' in df.columns:
            current_aqi = df['AQI'].dropna().iloc[-1]
        else:
            st.error("❌ No 'AQI' column found in dataset.")
            st.stop()

        def categorize_aqi(aqi):
            if aqi <= 50: return "Good", "green"
            elif aqi <= 100: return "Moderate", "yellow"
            elif aqi <= 200: return "Unhealthy (Sensitive)", "orange"
            elif aqi <= 300: return "Unhealthy", "red"
            elif aqi <= 400: return "Very Unhealthy", "purple"
            else: return "Hazardous", "maroon"

        aqi_status, color = categorize_aqi(current_aqi)

        st.markdown(f"### 🧾 Latest AQI Value: **{current_aqi:.1f} ({aqi_status})**")

        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            title={"text": f"Current AQI: {aqi_status}"},
            gauge={"axis": {"range": [0, 500]}, "bar": {"color": color}}
        ))
        st.plotly_chart(fig1, use_container_width=True)

        st.dataframe(df.tail(10), use_container_width=True)

    # ============================================================
    # TAB 2: Pollutant Concentrations vs WHO Limits
    # ============================================================
    with tab2:
        st.subheader("💨 Pollutant Concentrations vs WHO Limits")

        pollutants = ['PM2.5', 'PM10', 'O3']
        available = [p for p in pollutants if p in df.columns]

        if not available:
            st.warning("No pollutant columns (PM2.5, PM10, O3) found in dataset.")
        else:
            fig2 = go.Figure()
            for p in available:
                fig2.add_trace(go.Scatter(x=df[date_col], y=df[p], mode='lines', name=p))

            # WHO limits
            limits = {'PM2.5': 25, 'PM10': 50, 'O3': 100}
            for p, limit in limits.items():
                if p in available:
                    fig2.add_trace(go.Scatter(
                        x=df[date_col], y=[limit]*len(df),
                        mode='lines', name=f"{p} WHO Limit", line=dict(dash='dash', color='gray')
                    ))

            fig2.update_layout(title="Pollutant Trends vs WHO Limits", xaxis_title="Date", yaxis_title="Concentration (µg/m³)")
            st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # TAB 3: 7-Day Forecast (Demo)
    # ============================================================
    with tab3:
        st.subheader("📅 7-Day AQI Forecast (Demo Data)")

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        forecast_aqi = [45, 80, 130, 200, 160, 90, 50]
        forecast_df = pd.DataFrame({'Day': days, 'AQI': forecast_aqi})
        forecast_df['Category'] = forecast_df['AQI'].apply(lambda x: categorize_aqi(x)[0])
        forecast_df['Color'] = forecast_df['AQI'].apply(lambda x: categorize_aqi(x)[1])

        fig3 = px.bar(forecast_df, x='Day', y='AQI', color='Category', color_discrete_map={
            'Good': 'green', 'Moderate': 'yellow', 'Unhealthy (Sensitive)': 'orange',
            'Unhealthy': 'red', 'Very Unhealthy': 'purple', 'Hazardous': 'maroon'
        })
        fig3.update_layout(title="7-Day Forecasted AQI Levels", yaxis_title="AQI", xaxis_title="Day")
        st.plotly_chart(fig3, use_container_width=True)

    # ============================================================
    # TAB 4: Alerts
    # ============================================================
    with tab4:
        st.subheader("⚠️ Active Pollution Alerts")

        alerts = forecast_df[forecast_df['AQI'] > 150]
        if not alerts.empty:
            alert_days = ", ".join(alerts['Day'].tolist())
            st.error(f"🚨 High pollution alert for: **{alert_days}**")
        else:
            st.success("✅ No high pollution alerts this week!")

        st.write("### 🔍 Detailed Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

else:
    st.info("👆 Upload your air quality dataset to start the alert system.")
