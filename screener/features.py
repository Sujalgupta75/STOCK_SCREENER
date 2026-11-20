from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "Daily_Return", "MA_Ratio_10", "MA_Ratio_20", "EMA_Ratio_12",
    "RSI_14", "MACD", "MACD_Signal", "Momentum_5", "Momentum_20",
    "Volatility_20", "Volatility_60", "Volume_Change", "Volume_Z_20",
    "Intraday_Range", "Close_Position"
]

def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def build_features(raw: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build past-only features and forward-return label separately per ticker."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"]).reset_index(drop=True)
    chunks = []
    for ticker, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date").copy()
        close = _safe_numeric(g["Close"]).where(lambda s: s > 0)
        volume = _safe_numeric(g["Volume"]).clip(lower=0)
        daily = close.pct_change(fill_method=None)
        ma10 = close.rolling(10, min_periods=10).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(14, min_periods=14).mean()
        avg_loss = losses.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
        vol20 = daily.rolling(20, min_periods=20).std()
        vol60 = daily.rolling(60, min_periods=40).std()
        vol_mean = volume.rolling(20, min_periods=20).mean()
        vol_std = volume.rolling(20, min_periods=20).std()
        bar_range = (g["High"] - g["Low"]) / close
        denom = (g["High"] - g["Low"]).replace(0, np.nan)
        close_position = (close - g["Low"]) / denom
        g["Daily_Return"] = daily
        g["MA_Ratio_10"] = close / ma10 - 1
        g["MA_Ratio_20"] = close / ma20 - 1
        g["EMA_Ratio_12"] = close / ema12 - 1
        g["RSI_14"] = rsi
        g["MACD"] = macd / close
        g["MACD_Signal"] = macd_signal / close
        g["Momentum_5"] = close.pct_change(5, fill_method=None)
        g["Momentum_20"] = close.pct_change(20, fill_method=None)
        g["Volatility_20"] = vol20
        g["Volatility_60"] = vol60
        g["Volume_Change"] = volume.pct_change(fill_method=None).clip(-10, 10)
        g["Volume_Z_20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        g["Intraday_Range"] = bar_range
        g["Close_Position"] = close_position.clip(0, 1)
        future = close.shift(-horizon) / close - 1
        g["Future_Return"] = future
        g["Target"] = np.where(future.notna(), (future > 0).astype(float), np.nan)
        g["Target"] = g["Target"].astype(float)
        # Cap extreme values before imputation; keep NaN as missing.
        for col in FEATURES:
            g[col] = _safe_numeric(g[col]).clip(-1e4, 1e4)
        chunks.append(g)
    result = pd.concat(chunks, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan)

def data_quality_report(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    return {
        "rows": int(len(df)),
        "tickers": int(df["Ticker"].nunique()) if "Ticker" in df else 0,
        "date_start": str(df["Date"].min().date()) if len(df) else "—",
        "date_end": str(df["Date"].max().date()) if len(df) else "—",
        "duplicate_ticker_dates": int(df.duplicated(["Ticker", "Date"]).sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_numeric_values": int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0,
    }
