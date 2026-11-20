from __future__ import annotations
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from screener.data import download_live, demo_data
from screener.features import build_features, FEATURES, data_quality_report
from screener.modeling import train_and_evaluate, score_latest
from screener.charts import price_chart, feature_chart, correlation_chart, target_chart

APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="MarketLens | ML Stock Screener", page_icon="📈", layout="wide")
st.markdown("""
<style>
:root { --ink:#14243a; }
.block-container {padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1500px;}
.hero {padding: 1.45rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102b4e,#205b7d); color: white; margin-bottom: 1rem;}
.hero h1 {font-size:2.1rem; margin:0 0 .25rem 0; color:white;}
.hero p {margin:0; opacity:.88;}
div[data-testid="stMetric"] {background:rgba(120,145,170,.09); border:1px solid rgba(120,145,170,.22); padding:14px; border-radius:12px;}
.small-note {font-size:.86rem; opacity:.75;}
</style>
<div class="hero"><h1>MarketLens</h1><p>Machine Learning Based Trading & Investment Screener · Research dashboard</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis setup")
    mode = st.radio("Data source", ["Offline demo data", "Live Yahoo Finance"], index=0,
                    help="Offline mode is deterministic synthetic data. Live mode attempts to download real historical prices.")
    default_tickers = "AAPL, MSFT, NVDA, AMZN, GOOGL"
    ticker_text = st.text_area("Tickers (comma-separated)", value=default_tickers, height=90)
    tickers = list(dict.fromkeys([t.strip().upper() for t in ticker_text.replace("\n", ",").split(",") if t.strip()]))
    start = st.date_input("Start date", value=pd.Timestamp("2019-01-01").date())
    end = st.date_input("End date (inclusive)", value=pd.Timestamp.today().date())
    tune = st.checkbox("Tune model hyperparameters", value=True, help="Tuning uses time-series cross-validation and may take several minutes.")
    run = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Tip: start with 3–5 tickers and several years of history.")

if "raw" not in st.session_state:
    st.session_state.raw = None
    st.session_state.featured = None
    st.session_state.model_result = None
    st.session_state.source_note = ""

if run:
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()
    if start >= end:
        st.error("Start date must be earlier than end date.")
        st.stop()
    with st.spinner("Collecting and validating market data…"):
        try:
            if mode == "Live Yahoo Finance":
                raw, errors = download_live(tickers, start, (pd.Timestamp(end) + pd.Timedelta(days=1)).date())
            else:
                raw, errors = demo_data(tickers, start, end)
            st.session_state.raw = raw
            st.session_state.featured = build_features(raw, horizon=5)
            st.session_state.source_note = " · ".join(errors)
            st.session_state.model_result = None
        except Exception as exc:
            st.error(f"Data collection failed: {exc}")
            st.info("Switch to Offline demo data to verify that the app works without an external API.")
            st.stop()
    with st.spinner("Training models, performing time-series validation and evaluating the holdout period…"):
        try:
            st.session_state.model_result = train_and_evaluate(st.session_state.featured, tune=tune)
            payload = {
                "model": st.session_state.model_result["best_model"],
                "model_name": st.session_state.model_result["best_name"],
                "features": FEATURES,
                "trained_at": pd.Timestamp.now().isoformat(),
                "note": "Research model; validate before any real-world use."
            }
            joblib.dump(payload, ARTIFACT_DIR / "latest_model.joblib")
        except Exception as exc:
            st.session_state.model_result = None
            st.error(f"Model training/evaluation could not complete: {type(exc).__name__}: {exc}")
            st.warning("Try a longer date range, fewer tickers, or temporarily turn off hyperparameter tuning.")

raw = st.session_state.raw
featured = st.session_state.featured
result = st.session_state.model_result

if raw is None or featured is None:
    st.info("Choose your data source and tickers in the sidebar, then click **Run analysis**. Offline demo data is selected by default so the first run does not require a market-data API.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Pipeline", "Data → Features")
    c2.metric("Models", "3 candidates")
    c3.metric("Target", "Next 5 sessions")
    st.subheader("Product workflow")
    st.markdown("**Collect → Validate → Engineer features → Explore → Split by time → Tune → Evaluate → Rank → Export**")
    st.caption("Predictions are research estimates, not guarantees of future returns.")
    st.stop()

if st.session_state.source_note:
    st.warning(st.session_state.source_note)

quality = data_quality_report(featured)
m1,m2,m3,m4 = st.columns(4)
m1.metric("Valid market rows", f"{quality['rows']:,}")
m2.metric("Tickers", quality["tickers"])
m3.metric("Missing cells", f"{quality['missing_cells']:,}")
m4.metric("Duplicate ticker/date rows", quality["duplicate_ticker_dates"])

tab_overview, tab_eda, tab_models, tab_screener, tab_data = st.tabs(
    ["Overview", "EDA & Features", "Model Lab", "Stock Screener", "Data Explorer"]
)
with tab_overview:
    st.subheader("Market overview")
    st.plotly_chart(price_chart(raw), use_container_width=True)
    a,b,c = st.columns(3)
    a.metric("Date range", f"{pd.to_datetime(raw.Date).min().date()} → {pd.to_datetime(raw.Date).max().date()}")
    a.metric("Feature count", len(FEATURES))
    b.metric("Data source", "Live Yahoo Finance" if mode == "Live Yahoo Finance" else "Offline synthetic demo")
    b.metric("Forward label", "Positive return over next 5 sessions")
    c.metric("Infinite numeric values", quality["infinite_numeric_values"])
    c.metric("Tickers with data", raw["Ticker"].nunique())
    st.markdown("#### Latest observations")
    st.dataframe(raw.sort_values(["Ticker","Date"]).groupby("Ticker").tail(1)[["Ticker","Date","Close","Volume"]].sort_values("Ticker"), use_container_width=True, hide_index=True)

with tab_eda:
    st.subheader("Exploratory data analysis")
    tick = st.selectbox("Ticker for detailed chart", sorted(raw["Ticker"].unique()), key="eda_ticker")
    st.plotly_chart(feature_chart(featured, tick), use_container_width=True)
    left,right = st.columns(2)
    with left:
        st.plotly_chart(correlation_chart(featured.dropna(subset=FEATURES), FEATURES), use_container_width=True)
    with right:
        st.plotly_chart(target_chart(featured), use_container_width=True)
    st.markdown("#### Feature definitions")
    descriptions = {
        "Daily_Return":"One-session percentage return",
        "MA_Ratio_10":"Close relative to 10-session moving average",
        "MA_Ratio_20":"Close relative to 20-session moving average",
        "EMA_Ratio_12":"Close relative to 12-session exponential moving average",
        "RSI_14":"14-session relative strength index",
        "MACD":"MACD line scaled by close",
        "MACD_Signal":"MACD signal line scaled by close",
        "Momentum_5":"Five-session price momentum",
        "Momentum_20":"Twenty-session price momentum",
        "Volatility_20":"Rolling 20-session return standard deviation",
        "Volatility_60":"Rolling 60-session return standard deviation",
        "Volume_Change":"One-session volume change (clipped)",
        "Volume_Z_20":"Volume deviation from 20-session rolling mean",
        "Intraday_Range":"(High − Low) / Close",
        "Close_Position":"Close location within the daily high-low range",
    }
    st.dataframe(pd.DataFrame([{"Feature":k,"Meaning":v} for k,v in descriptions.items()]), use_container_width=True, hide_index=True)

with tab_models:
    st.subheader("Model comparison and holdout evaluation")
    if result is None:
        st.info("Model training did not finish successfully. Click Run analysis again, use more history, or disable tuning.")
    else:
        st.success(f"Selected model: {result['best_name']} — selected on validation data, not the test set.")
        rows = []
        for name, item in result["results"].items():
            vm = item.get("validation")
            rows.append({
                "Model":name,
                "Validation ROC-AUC": vm["roc_auc"] if vm else None,
                "Validation F1": vm["f1"] if vm else None,
                "CV ROC-AUC":item.get("cv_roc_auc"),
                "Status": "OK" if vm else item.get("error")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("#### Final held-out test metrics")
        tm = result["test_metrics"]
        cols = st.columns(5)
        for col, label, key in zip(cols, ["Accuracy","Precision","Recall","F1-score","ROC-AUC"], ["accuracy","precision","recall","f1","roc_auc"]):
            val = tm[key]
            col.metric(label, "N/A" if val is None else f"{val:.3f}")
        st.caption(f"Test period: {result['test_dates']['start']} to {result['test_dates']['end']} · {result['test_rows']:,} rows. Test data was held out chronologically.")
        cm = np.array(tm["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Predicted 0","Predicted 1"], y=["Actual 0","Actual 1"], color_continuous_scale="Blues", title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)
        if result["feature_importance"]:
            imp = pd.DataFrame(result["feature_importance"])
            fig2 = px.bar(imp.head(12).sort_values("importance"), x="importance", y="feature", orientation="h", title="Top feature importance / absolute coefficient")
            st.plotly_chart(fig2, use_container_width=True)
        with st.expander("Validation results and tuned parameters"):
            for name, item in result["results"].items():
                st.markdown(f"**{name}**")
                st.json({"best_params": item.get("best_params", {}), "cv_roc_auc": item.get("cv_roc_auc"), "error": item.get("error")})

with tab_screener:
    st.subheader("Latest stock ranking")
    if result is None:
        st.info("Train models first in the sidebar by clicking Run analysis.")
    else:
        ranked = score_latest(featured, result["best_model"], result["best_name"])
        display_cols = ["Ticker","Date","Close","Prediction_Probability","Signal","RSI_14","Momentum_5","Volatility_20"]
        st.caption("Ranks are based on model-estimated probability of a positive five-session return. They are research candidates, not personalized investment recommendations.")
        st.dataframe(
            ranked[display_cols].style.format({"Close":"{:.2f}","Prediction_Probability":"{:.1%}","RSI_14":"{:.1f}","Momentum_5":"{:.2%}","Volatility_20":"{:.2%}"}),
            use_container_width=True, hide_index=True
        )
        csv = ranked.to_csv(index=False).encode("utf-8")
        st.download_button("Download ranked stocks (CSV)", csv, file_name="stock_rankings.csv", mime="text/csv")
        st.plotly_chart(px.bar(ranked, x="Ticker", y="Prediction_Probability", color="Signal", title="Predicted probability by ticker", range_y=[0,1]), use_container_width=True)

with tab_data:
    st.subheader("Dataset and data-quality explorer")
    st.write("Raw OHLCV data")
    st.dataframe(raw.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download raw OHLCV CSV", raw.to_csv(index=False).encode("utf-8"), "market_data.csv", "text/csv")
    st.write("Engineered dataset")
    st.dataframe(featured.tail(500), use_container_width=True, hide_index=True)
    st.download_button("Download engineered dataset CSV", featured.to_csv(index=False).encode("utf-8"), "engineered_dataset.csv", "text/csv")
    st.markdown("#### Data quality report")
    st.json(quality)

st.divider()
st.caption("MarketLens · Educational Statistical ML project · Historical patterns do not guarantee future results.")
