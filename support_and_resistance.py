import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="📈 Stock Strategies Hub", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Helper functions --------------------
def get_country_suffix(country):
    country = country.lower()
    if country == "india":
        return ".NS"
    elif country == "australia":
        return ".AX"
    else:
        return ""

def get_close_series(df):
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series

# -------------------- Strategies --------------------
def buy_and_hold_strategy(stock_symbol, years, country):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")):
        stock_symbol += suffix

    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None

    df['Market Return'] = df['Close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1

    return df

def moving_average_crossover_strategy(stock_symbol, years, short_window, long_window, country):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")):
        stock_symbol += suffix

    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None

    df['MA_short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_long'] = df['Close'].rolling(window=long_window).mean()
    df['Signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1
    df['Position'] = df['Signal'].diff()

    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Signal'].shift(1)
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() - 1

    return df

def rsi_ma_stoploss_strategy(stock_symbol, years, investment_amount, short_ma, long_ma,
                             rsi_lower, rsi_upper, stoploss_pct, country):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")):
        stock_symbol += suffix

    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None, []

    close_series = get_close_series(df)

    df['RSI'] = RSIIndicator(close_series, window=14).rsi()
    df['SMA_short'] = SMAIndicator(close_series, window=short_ma).sma_indicator()
    df['SMA_long'] = SMAIndicator(close_series, window=long_ma).sma_indicator()

    df['Signal'] = 0
    df.loc[(df['RSI'] > rsi_lower) & (df['SMA_short'] > df['SMA_long']), 'Signal'] = 1

    position = 0
    entry_price = 0.0
    trades = 0
    positions_list = []
    trade_log = []

    for i in range(len(df)):
        date = df.index[i]
        price = close_series.iloc[i]
        if position == 0:
            if df['Signal'].iloc[i] == 1:
                position = 1
                entry_price = price
                trades += 1
                trade_log.append([date.date(), "Buy", price])
        else:
            if price <= entry_price * (1 - stoploss_pct):
                position = 0
                trades += 1
                trade_log.append([date.date(), "Sell (Stoploss)", price])
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position = 0
                trades += 1
                trade_log.append([date.date(), "Sell (Trend Reversal)", price])
        positions_list.append(position)

    df['Position'] = positions_list
    df['Market Return'] = close_series.pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position'].shift(1).fillna(0)
    df['Portfolio Value'] = investment_amount * (1 + df['Strategy Return']).cumprod()

    return df, trade_log

def rsi_ma_stoploss_backtest(stock_symbol, years, investment_amount, short_ma, long_ma,
                             rsi_lower, rsi_upper, stoploss_pct, country):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")):
        stock_symbol += suffix

    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None

    close_series = get_close_series(df)
    df['RSI'] = RSIIndicator(close_series, window=14).rsi()
    df['SMA_short'] = SMAIndicator(close_series, window=short_ma).sma_indicator()
    df['SMA_long'] = SMAIndicator(close_series, window=long_ma).sma_indicator()
    df['Signal'] = 0
    df.loc[(df['RSI'] > rsi_lower) & (df['SMA_short'] > df['SMA_long']), 'Signal'] = 1

    position = 0
    entry_price = 0.0
    trades = 0
    positions_list = []

    for i in range(len(df)):
        price = close_series.iloc[i]
        if position == 0:
            if df['Signal'].iloc[i] == 1: 
                position, entry_price, trades = 1, price, trades + 1
        else:
            if price <= entry_price * (1 - stoploss_pct):
                position, trades = 0, trades + 1
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position, trades = 0, trades + 1
        positions_list.append(position)

    df['Position'] = positions_list

    df['Market Return'] = close_series.pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position'].shift(1).fillna(0)
    df['Portfolio Value'] = investment_amount * (1 + df['Strategy Return']).cumprod()

    last_action_idx = df.index[(df['Position'] != df['Position'].shift(1).fillna(0))].max()
    last_action_price = close_series.loc[last_action_idx] if pd.notnull(last_action_idx) else np.nan

    return {
        'Ticker': stock_symbol,
        'Final Portfolio Value': round(df['Portfolio Value'].iloc[-1], 2),
        'Total Return (%)': round((df['Portfolio Value'].iloc[-1] / investment_amount - 1) * 100, 2),
        'Trades Executed': trades,
        'Last Action Date': str(last_action_idx.date()) if pd.notnull(last_action_idx) else "No trades",
        'Last Action Price': round(last_action_price, 2) if pd.notnull(last_action_idx) else "NA"
    }

# -------------------- UI --------------------
st.title("📊 Stock Strategies Hub")

choice = st.sidebar.selectbox(
    "Select Strategy",
    ["Buy & Hold", "Moving Average Crossover", "RSI+SMA+Stoploss (Single)", "RSI+SMA+Stoploss (Multi)"]
)

if choice == "Buy & Hold":
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol", "RELIANCE")
    with col2:
        country = st.selectbox("Country", ["India", "Australia", "US"])
    with col3:
