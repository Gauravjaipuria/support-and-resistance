import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from scipy.stats import linregress

st.set_page_config(page_title="Stock Strategies Hub", layout="wide")

# ----------- Helper: country suffix ------------
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

# ---------------- BUY & HOLD STRATEGY ----------------
def buy_and_hold_strategy(stock_symbol, years=3, country="India"):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")) and stock_symbol.isalpha():
        stock_symbol += suffix
    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None
    df['Market Return'] = df['Close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    return df

# ---------------- MOVING AVERAGE CROSSOVER ----------------
def moving_average_crossover_strategy(stock_symbol, years=3, short_window=20, long_window=50, country="India"):
    suffix = get_country_suffix(country)
    if not stock_symbol.endswith((".NS", ".AX")) and stock_symbol.isalpha():
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

# ---------------- RSI + SMA + STOPLOSS (SINGLE) ----------------
def rsi_ma_stoploss_strategy(stock_symbol, years=3, investment_amount=100000,
                             short_ma=20, long_ma=50, rsi_lower=30, rsi_upper=70,
                             stoploss_pct=0.01, country="India"):
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
    position = 0; entry_price = 0.0; trades = 0; positions_list = []; trade_log = []
    for i in range(len(df)):
        date = df.index[i]; price = close_series.iloc[i]
        if position == 0:
            if df['Signal'].iloc[i] == 1:
                position = 1; entry_price = price; trades += 1
                trade_log.append([date.date(), 'Buy', price])
        else:
            if price <= entry_price * (1 - stoploss_pct):
                position = 0; trades += 1
                trade_log.append([date.date(), 'Sell (Stoploss)', price])
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position = 0; trades += 1
                trade_log.append([date.date(), 'Sell (Trend Reversal)', price])
        positions_list.append(position)
    df['Position'] = positions_list
    df['Market Return'] = close_series.pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position'].shift(1).fillna(0)
    df['Portfolio Value'] = investment_amount * (1 + df['Strategy Return']).cumprod()
    return df, trade_log

# ---------------- MULTI STOCK RSI + SMA + STOPLOSS ----------------
def rsi_ma_stoploss_backtest(stock_symbol, years=3, investment_amount=100000,
                             short_ma=20, long_ma=50, rsi_lower=30, rsi_upper=70,
                             stoploss_pct=0.01, country="India"):
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
    position = 0; entry_price = 0.0; trades = 0; positions_list = []
    for i in range(len(df)):
        price = close_series.iloc[i]
        if position == 0:
            if df['Signal'].iloc[i] == 1: position, entry_price, trades = 1, price, trades+1
        else:
            if price <= entry_price * (1 - stoploss_pct):
                position, trades = 0, trades+1
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position, trades = 0, trades+1
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

# ---------------- BREAKOUT & SUPPORT ----------------
def support_resistance_analysis(ticker, country="India"):
    suffix = get_country_suffix(country)
    ticker += suffix
    stock = yf.Ticker(ticker)
    hist_6mo = stock.history(period="6mo")
    hist_1y = stock.history(period="1y")
    if hist_6mo.empty:
        return None
    r1y_min = hist_1y['Close'].min()
    r1y_max = hist_1y['Close'].max()
    recent = hist_6mo.iloc[-1]
    high, low, close = recent['High'], recent['Low'], recent['Close']
    pivot = (high + low + close) / 3
    s1 = (2 * pivot) - high
    r1 = (2 * pivot) - low
    return {
        "Ticker": ticker,
        "Support 1": round(s1,2),
        "Resistance 1": round(r1,2),
        "1Y Low": round(r1y_min,2),
        "1Y High": round(r1y_max,2)
    }

# ------------------- STREAMLIT UI -------------------
st.sidebar.title("Choose Strategy")
choice = st.sidebar.selectbox("Select", ["Buy & Hold", "Moving Average Crossover", "RSI+SMA+Stoploss (Single)", "RSI+SMA+Stoploss (Multi)", "Support/Resistance"])

if choice == "Buy & Hold":
    symbol = st.text_input("Stock Symbol", "RELIANCE")
    country = st.selectbox("Country", ["India", "Australia", "US"], key="bnh_country")
    years = st.number_input("Years", 1, 15, 3)
    if st.button("Run Strategy"):
        df = buy_and_hold_strategy(symbol, years, country)
        if df is not None:
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Cumulative Market Return'], label="Buy & Hold Return")
            ax.legend(); ax.grid()
            st.pyplot(fig)

elif choice == "Moving Average Crossover":
    symbol = st.text_input("Stock Symbol", "RELIANCE")
    country = st.selectbox("Country", ["India", "Australia", "US"], key="mac_country")
    years = st.number_input("Years", 1, 15, 3)
    s_win = st.number_input("Short MA", 5, 100, 20)
    l_win = st.number_input("Long MA", 10, 200, 50)
    if st.button("Run Strategy"):
        df = moving_average_crossover_strategy(symbol, years, s_win, l_win, country)
        if df is not None:
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Cumulative Market Return'], label="Market")
            ax.plot(df.index, df['Cumulative Strategy Return'], label="Strategy")
            ax.legend(); ax.grid()
            st.pyplot(fig)

elif choice == "RSI+SMA+Stoploss (Single)":
    symbol = st.text_input("Stock Symbol", "RELIANCE")
    country = st.selectbox("Country", ["India", "Australia", "US"], key="singlerm_country")
    years = st.number_input("Years", 1, 15, 3)
    invest = st.number_input("Investment", 1000, 10000000, 100000)
    if st.button("Run Strategy"):
        df, log = rsi_ma_stoploss_strategy(symbol, years, invest, country=country)
        if df is not None:
            st.write(pd.DataFrame(log, columns=["Date", "Action", "Price"]))
            st.line_chart(df['Portfolio Value'])

elif choice == "RSI+SMA+Stoploss (Multi)":
    symbols = st.text_input("Symbols comma-separated", "RELIANCE,TCS")
    country = st.selectbox("Country", ["India", "Australia", "US"], key="multirm_country")
    years = st.number_input("Years", 1, 15, 3)
    invest = st.number_input("Investment", 1000, 10000000, 100000)
    if st.button("Run Backtest"):
        results = []
        for s in symbols.split(","):
            res = rsi_ma_stoploss_backtest(s.strip().upper(), years, invest, country=country)
            if res:
                results.append(res)
        if results: st.dataframe(pd.DataFrame(results))

elif choice == "Support/Resistance":
    symbols = st.text_input("Symbols", "RELIANCE,TCS")
    country = st.selectbox("Country", ["India", "Australia", "US"], key="sr_country")
    if st.button("Analyze"):
        results = []
        for s in symbols.split(","):
            r = support_resistance_analysis(s.strip().upper(), country)
            if r:
                results.append(r)
        if results: st.dataframe(pd.DataFrame(results))
