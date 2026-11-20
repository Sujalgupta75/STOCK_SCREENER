from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

def _normalise_yf_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance single- or multi-index columns into OHLCV columns."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance may return (Price, Ticker) or (Ticker, Price).
        levels = [set(map(str, out.columns.get_level_values(i))) for i in range(out.columns.nlevels)]
        ticker_level = next((i for i, vals in enumerate(levels) if ticker in vals), None)
        if ticker_level is not None:
            try:
                out = out.xs(ticker, axis=1, level=ticker_level, drop_level=True)
            except Exception:
                pass
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(c[0]) for c in out.columns]
    out.columns = [str(c).title() for c in out.columns]
    # If the source uses Adj Close only, use it as Close.
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    keep = [c for c in REQUIRED if c in out.columns]
    if "Close" not in keep:
        return pd.DataFrame()
    for c in ["Open", "High", "Low"]:
        if c not in out.columns:
            out[c] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[REQUIRED].copy()
    out["Ticker"] = ticker
    out.index = pd.to_datetime(out.index, errors="coerce")
    out.index.name = "Date"
    return out.reset_index()

def download_live(tickers, start, end=None):
    import yfinance as yf
    frames, errors = [], []
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker, start=str(start), end=(str(end) if end else None),
                auto_adjust=True, progress=False, threads=False
            )
            clean = _normalise_yf_columns(raw, ticker)
            if clean.empty:
                errors.append(f"{ticker}: no usable rows returned")
            else:
                frames.append(clean)
        except Exception as exc:
            errors.append(f"{ticker}: {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError("No live data could be downloaded. " + " | ".join(errors))
    data = pd.concat(frames, ignore_index=True)
    return validate_ohlcv(data), errors

def demo_data(tickers, start="2019-01-01", end=None, rows_per_ticker=1400):
    """Generate deterministic synthetic OHLCV data for offline demonstration."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start_dt, end_dt)
    if len(dates) < 100:
        dates = pd.bdate_range(end=end_dt, periods=max(100, rows_per_ticker))
    frames = []
    for ticker in tickers:
        seed = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n = len(dates)
        drift = rng.normal(0.00025, 0.00012)
        returns = drift + rng.normal(0, 0.014, n)
        close = (60 + rng.uniform(20, 180)) * np.exp(np.cumsum(returns))
        overnight = rng.normal(0, 0.004, n)
        open_ = close * (1 + overnight)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, n))
        volume = rng.lognormal(mean=14.2, sigma=0.35, size=n).astype(float)
        frames.append(pd.DataFrame({
            "Date": dates, "Open": open_, "High": high, "Low": low,
            "Close": close, "Volume": volume, "Ticker": ticker
        }))
    return validate_ohlcv(pd.concat(frames, ignore_index=True)), ["Offline demo data: synthetic, not actual market prices."]

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in REQUIRED:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out = out.dropna(subset=["Date", "Ticker", "Close"])
    out = out[out["Close"] > 0]
    out = out[out["Volume"].fillna(0) >= 0]
    # Repair missing OHLC values conservatively from Close.
    for c in ["Open", "High", "Low"]:
        out[c] = out[c].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0)
    # Drop impossible bars rather than silently accepting them.
    valid_range = (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) & (
        out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)
    )
    out = out[valid_range]
    out = out.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"], keep="last")
    if out.empty:
        raise ValueError("No valid OHLCV rows remain after validation.")
    return out.reset_index(drop=True)
