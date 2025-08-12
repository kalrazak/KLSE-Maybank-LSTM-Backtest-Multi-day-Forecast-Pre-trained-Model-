# app.py
# Streamlit: Load pre-trained Maybank (KLSE: 1155.KL) multi-feature LSTM model for backtest & forecast
# Educational use only – not financial advice.

import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="KLSE Maybank – LSTM Forecast",
    page_icon="📈",
    layout="wide",
)

# ---------------------------
# Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_data_with_indicators(ticker: str, start="2010-01-01"):
    """Fetch price data and add SMA, MACD, RSI."""
    end = dt.datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # If multi-index columns (happens for some tickers), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    # Ensure Close exists
    if "Close" not in df.columns:
        st.error("No 'Close' column found. Check ticker symbol.")
        st.stop()

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").astype(float)

    # SMA
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()

    # MACD
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df

def recursive_forecast(last_seq_scaled: np.ndarray, model, scaler, lookback: int, days: int):
    """Forecast multiple future days recursively using LSTM."""
    seq = last_seq_scaled.copy()
    preds_scaled = []
    for _ in range(days):
        X = seq.reshape(1, lookback, seq.shape[1])
        p = model.predict(X, verbose=0)[0][0]
        # Append prediction with previous features
        new_row = seq[-1].copy()
        new_row[0] = p  # replace Close with prediction
        preds_scaled.append(p)
        seq = np.vstack([seq[1:], new_row])
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    # Pad with zeros for other features to inverse_transform
    pad = np.zeros((preds_scaled.shape[0], seq.shape[1] - 1))
    preds_full = np.hstack([preds_scaled, pad])
    preds = scaler.inverse_transform(preds_full)[:, 0]
    return preds

def compute_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.mean(np.abs((y_true - y_pred) / y_true)) * 100).item()
    return rmse, mae, mape

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="1155.KL")
lookback = st.sidebar.slider("Lookback (days)", 30, 180, 60, 5)
forecast_days = st.sidebar.slider("Forecast horizon (business days)", 1, 14, 7)
models_dir = st.sidebar.text_input("Models folder", value="models")

st.sidebar.markdown("---")
st.sidebar.caption("© For learning/portfolio. Not investment advice.")

# ---------------------------
# Main
# ---------------------------
st.title("📈 KLSE Maybank – LSTM Backtest & Multi-day Forecast (Pre-trained Model)")

# 1) Load Data with Technical Indicators
with st.spinner("Downloading historical data & computing indicators..."):
    df = fetch_data_with_indicators(ticker)
latest_close = float(df["Close"].iloc[-1])
st.success(f"Loaded {len(df)} rows with features. Latest close: {latest_close:.2f} MYR")

# 2) Paths to Saved Model & Scaler
model_path = os.path.join(models_dir, f"{ticker.replace('.', '_')}_lstm.h5")
scaler_path = os.path.join(models_dir, f"{ticker.replace('.', '_')}_scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model or scaler file not found! Please place them in the models folder.")
    st.stop()

# 3) Load Model & Scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)
st.success("✅ Loaded pre-trained LSTM model & scaler.")

# 4) Prepare Backtest Data (from 2021 onwards)
backtest_start = "2021-01-01"
test_df = df[df.index >= backtest_start].copy()

scaled_test = scaler.transform(test_df.values)

# Create sequences for backtest
X_test, y_test = [], []
for i in range(lookback, len(scaled_test)):
    X_test.append(scaled_test[i - lookback:i])
    y_test.append(scaled_test[i, 0])  # Close price
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1, 1)

# 5) Run Backtest
with st.spinner("Running backtest..."):
    preds_test_scaled = model.predict(X_test, verbose=0)

# Inverse transform predictions
pad_preds = np.zeros((preds_test_scaled.shape[0], scaled_test.shape[1] - 1))
pad_actual = np.zeros((y_test.shape[0], scaled_test.shape[1] - 1))
preds_test = scaler.inverse_transform(np.hstack([preds_test_scaled, pad_preds]))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack([y_test, pad_actual]))[:, 0]

rmse, mae, mape = compute_metrics(y_test_actual, preds_test)

# 6) Forecast Future
last_seq_scaled = scaled_test[-lookback:]
future_preds = recursive_forecast(last_seq_scaled, model, scaler, lookback, forecast_days)
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds.flatten()})

# 7) Display Metrics & Forecast
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader(f"Backtest Metrics (From {backtest_start})")
    st.table(pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "MAPE (%)": [mape]}).round(4))
with col2:
    st.subheader(f"{forecast_days}-Day Forecast")
    st.dataframe(forecast_df.style.format({"Predicted Close": "{:.4f}"}), use_container_width=True)

# ---------------------------
# 8) Plot - Interactive Plotly Chart
# ---------------------------
import plotly.graph_objects as go

st.subheader("📊 Price Chart: Historical, Backtest, Forecast")

# Define backtest dates
backtest_dates = test_df.index[lookback:]

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"], mode="lines", name="Historical Price",
    line=dict(color="#1f77b4", width=2)
))

# Backtest actual
fig.add_trace(go.Scatter(
    x=backtest_dates, y=y_test_actual, mode="lines", name="Actual (Backtest)",
    line=dict(color="#2ca02c", width=2)
))

# Backtest predicted
fig.add_trace(go.Scatter(
    x=backtest_dates, y=preds_test, mode="lines", name="Predicted (Backtest)",
    line=dict(color="#ff7f0e", dash="dash", width=2)
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_df["Date"], y=forecast_df["Predicted Close"], mode="lines+markers",
    name=f"{forecast_days}-Day Forecast", line=dict(color="#d62728", dash="dot", width=2)
))

# Layout with range selector & slider
fig.update_layout(
    title=f"{ticker} – LSTM Backtest & {forecast_days}-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Close Price (MYR)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 9) Download Button
st.download_button(
    "Download Forecast CSV",
    forecast_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker.replace('.', '_')}_forecast_{forecast_days}d.csv",
    mime="text/csv",
)
