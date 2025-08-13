import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from scipy.stats import linregress
import os
import zipfile

st.set_page_config(page_title="Stock Strategies Hub", layout="wide")

# ---------------- BUY & HOLD STRATEGY ----------------
def buy_and_hold_strategy(stock_symbol, years=3):
    if not stock_symbol.endswith(".NS") and stock_symbol.isalpha():
        stock_symbol += ".NS"
    df = yf.download(stock_symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.error(f"No data found for {stock_symbol}")
        return None
    df['Market Return'] = df['Close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    return df

