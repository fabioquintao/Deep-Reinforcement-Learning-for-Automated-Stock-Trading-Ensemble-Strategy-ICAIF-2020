import pandas as pd
import numpy as np
import yfinance as yf
from stockstats import StockDataFrame as Sdf
from bitcoin.config import TICKER, START_DATE, END_DATE


def download_btc(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    df = yf.download(TICKER, start=start, end=end, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    stock = Sdf.retype(df.copy())
    df["macd"] = stock["macd"]
    df["rsi"] = stock["rsi_14"]
    df["cci"] = stock["cci_20"]
    df["adx"] = stock["dx_14"]
    df["boll_ub"] = stock["boll_ub"]
    df["boll_lb"] = stock["boll_lb"]
    df["atr"] = stock["atr"]
    df = df.fillna(method="bfill")
    return df


def load_data(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    df = download_btc(start, end)
    df = add_indicators(df)
    return df


def data_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= start) & (df.index < end)
    out = df.loc[mask].copy()
    out = out.reset_index(drop=False)   # keep 'date' as column
    out.index = range(len(out))
    return out
