import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.title("📊 Stock Technical Analysis Dashboard")

tickers_input = st.text_input("Enter tickers (comma-separated)", "TCS, RELIANCE")
country = st.selectbox("Select Country", ["India", "Australia", "US"])

suffix = ".NS" if country == "India" else ".AX" if country == "Australia" else ""
tickers = [ticker.strip().upper() + suffix for ticker in tickers_input.split(",")]

results = []

for ticker in tickers:
    st.header(f"📈 {ticker}")
    stock = yf.Ticker(ticker)
    hist_6mo = stock.history(period="6mo")
    hist_1y = stock.history(period="1y")
    info = stock.info

    if hist_6mo.empty or hist_1y.empty or 'Close' not in hist_6mo:
        st.warning(f"No data for {ticker}")
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

    rsi = round(hist['RSI'].dropna().iloc[-1], 2) if not hist['RSI'].dropna().empty else None
    macd = round(hist['MACD'].dropna().iloc[-1], 2) if not hist['MACD'].dropna().empty else None
    adx = round(hist['ADX'].dropna().iloc[-1], 2) if not hist['ADX'].dropna().empty else None
    vwap_now = round(hist['VWAP'].iloc[-1], 2)
    rvol = round(hist['Volume'].iloc[-1] / hist['20-Day Avg Volume'].iloc[-1], 2) if hist['20-Day Avg Volume'].iloc[-1] != 0 else None
    obv_now = hist['OBV'].dropna().iloc[-1] if not hist['OBV'].dropna().empty else None
    obv_prev = hist['OBV'].dropna().iloc[-2] if len(hist['OBV'].dropna()) > 1 else None
    vol_spike = hist['Volume Spike'].iloc[-1]
    price_prev = hist['Close'].iloc[-2] if len(hist) > 1 else close

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

    if rsi is not None:
        if rsi < 40:
            rsi_signal = "Buy"
        elif rsi > 60:
            rsi_signal = "Sell"
        else:
            rsi_signal = "Hold"
    else:
        rsi_signal = "No Signal"

    if adx is not None:
        if adx > 25:
            trend_strength = "Strong Trend"
        elif adx < 20:
            trend_strength = "Weak/Sideways"
        else:
            trend_strength = "Moderate"
    else:
        trend_strength = "Unknown"

    vwap_bias = "Bullish Bias" if close > vwap_now else "Bearish Bias"

    short_term_risk = "Neutral"
    if (abs(close - price_prev) / price_prev) > 0.05 and vol_spike:
        short_term_risk = "High Volatility"
    elif vol_spike:
        short_term_risk = "Likely Speculation"
    elif rsi and (rsi > 70 or rsi < 30):
        short_term_risk = "RSI Extreme"

    if volume_signal == "Accumulation" and short_term_risk == "Neutral" and vwap_bias == "Bullish Bias" and trend_strength == "Strong Trend":
        investor_signal = "Strong Long-Term Buy Setup"
    elif short_term_risk != "Neutral":
        investor_signal = "Short-Term Volatility"
    elif vwap_bias == "Bearish Bias" and trend_strength == "Weak/Sideways":
        investor_signal = "Avoid - Weak Setup"
    else:
        investor_signal = "Unclear"

    last_30_vol = hist['Volume'].tail(30).reset_index(drop=True)
    x_vals = range(len(last_30_vol))
    slope, intercept, *_ = linregress(x_vals, last_30_vol)
    vol_trend = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Flat"

    st.markdown(f"""
    **Price**: {close}  
    **52W High**: {info.get("fiftyTwoWeekHigh")}  
    **52W Low**: {info.get("fiftyTwoWeekLow")}  
    **Support 1 / 2**: {round(s1, 2)} / {round(s2, 2)}  
    **Resistance 1 / 2**: {round(r1, 2)} / {round(r2, 2)}  
    **1Y Support / Resistance**: {support_1y} / {resistance_1y}  
    **P/E Ratio**: {info.get("trailingPE")}  
    **Beta**: {info.get("beta")}  
    **RSI**: {rsi} ({rsi_signal})  
    **MACD**: {macd}  
    **ADX**: {adx} ({trend_strength})  
    **VWAP**: {vwap_now} ({vwap_bias})  
    **Relative Volume**: {rvol}  
    **OBV**: {round(obv_now, 2) if obv_now else None}  
    **Volume Signal**: {volume_signal}  
    **Volatility**: {short_term_risk}  
    **Investor Signal**: **{investor_signal}**  
    **Volume Trend (30D)**: {vol_trend}  
    """)

    st.line_chart(hist['Close'].tail(90))
    st.bar_chart(hist['Volume'].tail(30))
