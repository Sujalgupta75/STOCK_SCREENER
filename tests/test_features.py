import numpy as np
import pandas as pd
from screener.data import validate_ohlcv, demo_data
from screener.features import build_features, FEATURES, data_quality_report

def test_demo_data_and_feature_engineering():
    raw, notes = demo_data(["AAA", "BBB"], "2021-01-01", "2024-12-31")
    assert not raw.empty
    assert set(["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]).issubset(raw.columns)
    featured = build_features(raw, horizon=5)
    assert set(FEATURES).issubset(featured.columns)
    assert "Target" in featured.columns
    assert featured.groupby("Ticker").size().min() > 100
    assert data_quality_report(featured)["infinite_numeric_values"] == 0

def test_target_is_forward_return_sign():
    dates = pd.bdate_range("2020-01-01", periods=100)
    close = np.arange(100, 200, dtype=float)
    raw = pd.DataFrame({
        "Date": dates, "Ticker": "AAA", "Open": close, "High": close + 1,
        "Low": close - 1, "Close": close, "Volume": np.arange(1,101)*1000
    })
    featured = build_features(raw, horizon=5)
    row = featured.iloc[30]
    assert row["Target"] == 1
    assert np.isclose(row["Future_Return"], close[35]/close[30]-1)

def test_validation_deduplicates_and_removes_bad_close():
    dates = pd.bdate_range("2020-01-01", periods=3)
    df = pd.DataFrame({
        "Date": list(dates)+[dates[0]], "Ticker": ["AAA"]*4,
        "Open":[10,11,12,10], "High":[11,12,13,11], "Low":[9,10,11,9],
        "Close":[10,11,12,10], "Volume":[100,100,100,100]
    })
    clean = validate_ohlcv(df)
    assert len(clean) == 3
    assert clean.duplicated(["Ticker","Date"]).sum() == 0
