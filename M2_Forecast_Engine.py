# =============================================================
# 🌤️ Air Quality Forecast Engine — Enhanced (Final Fixed Version)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# =============================================================
# Streamlit Page Config
# =============================================================
st.set_page_config(
    page_title="Air Quality Forecast Engine — Enhanced",
    layout="wide",
    page_icon="🌤️",
)

st.title("🌤️ Air Quality Forecast Engine — Enhanced")
st.markdown("Compare forecasting methods, compute metrics, and generate short-term forecasts for pollutants.")

# =============================================================
# Sidebar Filters
# =============================================================
st.sidebar.header("Filters")

# Upload or use sample
uploaded_file = st.sidebar.file_uploader("Upload your Air Quality CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV loaded ✅")
else:
    st.sidebar.info("Use sample dataset (demo)")
    df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=120),
        "City": np.random.choice(["Delhi", "Ahmedabad", "Mumbai"], 120),
        "PM2.5": np.random.uniform(40, 200, 120)
    })

# =============================================================
# Sidebar Filters: Location & Pollutant
# =============================================================
cities = ["All"] + sorted(df["City"].unique().tolist())
selected_city = st.sidebar.selectbox("Location", cities)

pollutants = ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"] if "PM2.5" in df.columns else df.columns[1:]
selected_pollutant = st.sidebar.selectbox("Pollutant to analyze/forecast", pollutants)

forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7)
train_fraction = st.sidebar.slider("Train fraction (for backtesting)", 0.3, 0.9, 0.55)

models_selected = st.sidebar.multiselect("Models to evaluate & run", ["ARIMA", "Naive", "Moving Average"], default=["ARIMA"])

if st.sidebar.button("Apply Filters & Run"):
    st.session_state["run_forecast"] = True

# =============================================================
# Filter and Prepare Data
# =============================================================
if selected_city != "All":
    df = df[df["City"] == selected_city]

st.markdown("### 🧾 Filtered Data Preview")
st.dataframe(df.head(50), use_container_width=True)

# =============================================================
# Tabs for Navigation
# =============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "⚙️ Models & Backtest", "📈 Forecast", "💡 Insights", "⬇️ Export"])

with tab1:
    st.write("### Data Summary")
    st.write(df.describe())

# =============================================================
# If Forecast Button Clicked
# =============================================================
if "run_forecast" in st.session_state and st.session_state["run_forecast"]:

    with tab2:
        st.subheader("⚙️ Model Training & Backtesting")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(subset="Date", keep="first")
        df = df.sort_values("Date")

        df = df.rename(columns={selected_pollutant: "y"})
        series = df.set_index("Date")["y"].asfreq('D')
        series = series.interpolate(method='time')

        # Split data
        train_size = int(len(series) * train_fraction)
        train, test = series[:train_size], series[train_size:]

        results = {}

        # =============================================================
        # ARIMA Model
        # =============================================================
        if "ARIMA" in models_selected:
            try:
                model = ARIMA(train, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
                mae = mean_absolute_error(test, forecast)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                results["ARIMA"] = {"MAE": mae, "RMSE": rmse, "Forecast": forecast}
                st.success(f"✅ ARIMA model trained successfully! (MAE={mae:.2f}, RMSE={rmse:.2f})")
            except Exception as e:
                st.error(f"ARIMA failed: {e}")

        # =============================================================
        # Naive Forecast
        # =============================================================
        if "Naive" in models_selected:
            forecast = np.repeat(train.iloc[-1], len(test))
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            results["Naive"] = {"MAE": mae, "RMSE": rmse, "Forecast": forecast}
            st.info(f"Naive Forecast → MAE={mae:.2f}, RMSE={rmse:.2f}")

        # =============================================================
        # Moving Average Forecast
        # =============================================================
        if "Moving Average" in models_selected:
            forecast = np.repeat(train.rolling(window=3).mean().iloc[-1], len(test))
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            results["Moving Average"] = {"MAE": mae, "RMSE": rmse, "Forecast": forecast}
            st.warning(f"Moving Average → MAE={mae:.2f}, RMSE={rmse:.2f}")

        st.write("### 📉 Backtest Results")
        st.dataframe(pd.DataFrame(results).T)

    # =============================================================
    # Forecast Tab
    # =============================================================
    with tab3:
        st.subheader("📈 Forecast Visualization")

        best_model = min(results, key=lambda k: results[k]["RMSE"])
        st.success(f"🏆 Best Model: {best_model}")

        test_index = test.index
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train.index, train, label="Train", color="blue")
        ax.plot(test_index, test, label="Actual", color="gray")
        ax.plot(test_index, results[best_model]["Forecast"], label=f"Forecast ({best_model})", color="red")
        ax.legend()
        st.pyplot(fig)

        # Extend forecast
        model = ARIMA(series, order=(1, 1, 1)).fit()
        future_forecast = model.forecast(steps=forecast_horizon)
        future_dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_horizon)
        future_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast})
        st.write("### 🔮 Future Forecast")
        st.dataframe(future_df, use_container_width=True)

    # =============================================================
    # Insights Tab
    # =============================================================
    with tab4:
        st.subheader("💡 Model Insights")
        for model_name, metrics in results.items():
            st.markdown(f"**{model_name}** → MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

    # =============================================================
    # Export Tab
    # =============================================================
    with tab5:
        st.subheader("⬇️ Export Results")
        csv = future_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast CSV", csv, "forecast_results.csv", "text/csv")
        st.success("Export complete ✅")
