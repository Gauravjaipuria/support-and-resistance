import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("📊 Stock Technical Analysis Dashboard")

# --- User Input ---
tickers_input = st.text_input("Enter Tickers (comma-separated):", "TCS, INFY")
country = st.selectbox("Select Country:", ["India", "Australia", "US"])

suffix = ".NS" if country == "India" else ".AX" if country == "Australia" else ""
tickers = [ticker.strip().upper() + suffix for ticker in tickers_input.split(",")]

# Results list
results = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist_6mo = stock.history(period="6mo")
    hist_1y = stock.history(period="1y")
    info = stock.info

    if hist_6mo.empty or hist_1y.empty or 'Close' not in hist_6mo:
        st.warning(f"No data found for {ticker}")
        continue

    hist = hist_6mo.copy()
    hist['RSI'] = ta.momentum.RSIIndicator(close=hist['Close']).rsi()
    hist['MACD'] = ta.trend.MACD(close=hist['Close']).macd()
    hist['OBV'] = ta.volume.OnBalanceVolumeIndicator(hist['Close'], hist['Volume']).on_balance_volume()
    hist['ADX'] = ta.trend.ADXIndicator(hist['High'], hist['Low'], hist['Close']).adx()
    hist['VWAP'] = (hist['Volume'] * (hist['High'] + hist['Low'] + hist['Close']) / 3).cumsum() / hist['Volume'].cumsum()
    hist['20-Day Avg Volume'] = hist['Volume'].rolling(20).mean()
    hist['Volume Spike'] = hist['Volume'] > 2 * hist['20-Day Avg Volume']

    recent = hist.iloc[-1]
    high, low, close = recent['High'], recent['Low'], recent['Close']
    pivot = (high + low + close) / 3
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)

    rsi = round(hist['RSI'].dropna().iloc[-1], 2)
    macd = round(hist['MACD'].dropna().iloc[-1], 2)
    adx = round(hist['ADX'].dropna().iloc[-1], 2)
    vwap_now = round(hist['VWAP'].iloc[-1], 2)
    rvol = round(hist['Volume'].iloc[-1] / hist['20-Day Avg Volume'].iloc[-1], 2)
    obv_now = hist['OBV'].dropna().iloc[-1]
    obv_prev = hist['OBV'].dropna().iloc[-2]
    vol_spike = hist['Volume Spike'].iloc[-1]
    price_prev = hist['Close'].iloc[-2]

    support_1y = round(hist_1y['Close'].min(), 2)
    resistance_1y = round(hist_1y['Close'].max(), 2)

    if vol_spike and close > price_prev and obv_now > obv_prev:
        volume_signal = "Accumulation"
    elif vol_spike and close < price_prev:
        volume_signal = "Distribution"
    elif vol_spike and abs(close - price_prev) < 0.5:
        volume_signal = "Absorption"
    else:
        volume_signal = "Neutral"

    rsi_signal = "Buy" if rsi < 40 else "Sell" if rsi > 60 else "Hold"
    trend_strength = "Strong Trend" if adx > 25 else "Weak/Sideways" if adx < 20 else "Moderate"
    vwap_bias = "Bullish Bias" if close > vwap_now else "Bearish Bias"

    short_term_risk = "Neutral"
    if (abs(close - price_prev) / price_prev) > 0.05 and vol_spike:
        short_term_risk = "High Volatility"
    elif vol_spike:
        short_term_risk = "Likely Speculation"
    elif rsi > 70 or rsi < 30:
        short_term_risk = "RSI Extreme"

    investor_signal = "Strong Long-Term Buy Setup" if (volume_signal == "Accumulation" and short_term_risk == "Neutral" and vwap_bias == "Bullish Bias" and trend_strength == "Strong Trend") else "Short-Term Volatility" if short_term_risk != "Neutral" else "Avoid - Weak Setup" if (vwap_bias == "Bearish Bias" and trend_strength == "Weak/Sideways") else "Unclear"

    # Volume Trend
    last_30_vol = hist['Volume'].tail(30).reset_index(drop=True)
    x_vals = range(len(last_30_vol))
    slope, intercept, *_ = linregress(x_vals, last_30_vol)
    vol_trend = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Flat"

    st.subheader(f"📌 {ticker}")
    st.write({
        "Current Price": close,
        "52W High": info.get("fiftyTwoWeekHigh"),
        "52W Low": info.get("fiftyTwoWeekLow"),
        "Support 1": round(s1, 2),
        "Support 2": round(s2, 2),
        "Resistance 1": round(r1, 2),
        "Resistance 2": round(r2, 2),
        "Support 1Y": support_1y,
        "Resistance 1Y": resistance_1y,
        "P/E Ratio": info.get("trailingPE"),
        "Beta": info.get("beta"),
        "RSI": rsi,
        "MACD": macd,
        "ADX": adx,
        "VWAP": vwap_now,
        "Relative Volume": rvol,
        "OBV": round(obv_now, 2),
        "RSI Signal": rsi_signal,
        "Volume Signal": volume_signal,
        "Trend Strength": trend_strength,
        "VWAP Bias": vwap_bias,
        "Volatility Flag": short_term_risk,
        "Investor Signal": investor_signal,
        "Volume Trend (30D)": vol_trend
    })

    # --- Charts ---
    st.write("### 📈 Price & Support/Resistance")
    fig, ax = plt.subplots(figsize=(10, 4))
    hist['Close'].tail(90).plot(ax=ax, color='blue', label='Close')
    ax.axhline(s1, color='green', linestyle='--', label=f'Support 1: {round(s1,2)}')
    ax.axhline(s2, color='green', linestyle=':', label=f'Support 2: {round(s2,2)}')
    ax.axhline(r1, color='red', linestyle='--', label=f'Resistance 1: {round(r1,2)}')
    ax.axhline(r2, color='red', linestyle=':', label=f'Resistance 2: {round(r2,2)}')
    ax.axhline(support_1y, color='purple', linestyle='-', label=f'1Y Support: {support_1y}')
    ax.axhline(resistance_1y, color='orange', linestyle='-', label=f'1Y Resistance: {resistance_1y}')
    ax.set_title(f'{ticker} - 90 Day Price Trend')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.write("### 🔄 Volume Trend (30 Days)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(x_vals, last_30_vol, color='skyblue', label='Volume')
    trend_line = intercept + slope * pd.Series(x_vals)
    ax2.plot(x_vals, trend_line, color='red', linestyle='--', label='Trend Line')
    ax2.set_title('30-Day Volume Trend')
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)
