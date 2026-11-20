import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def price_chart(df):
    fig = go.Figure()
    for ticker, g in df.groupby("Ticker"):
        fig.add_trace(go.Scatter(x=g["Date"], y=g["Close"], mode="lines", name=ticker))
    fig.update_layout(title="Closing price history", xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=420)
    return fig

def feature_chart(df, ticker):
    g = df[df["Ticker"] == ticker].sort_values("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["Date"], y=g["Close"], mode="lines", name="Close"))
    if "MA_10" in g:
        fig.add_trace(go.Scatter(x=g["Date"], y=g["MA_10"], mode="lines", name="SMA 10"))
    if "MA_20" in g:
        fig.add_trace(go.Scatter(x=g["Date"], y=g["MA_20"], mode="lines", name="SMA 20"))
    fig.update_layout(title=f"{ticker}: price and moving averages", height=380, hovermode="x unified")
    return fig

def correlation_chart(df, features):
    corr = df[features].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title="Feature correlation matrix")
    fig.update_layout(height=520)
    return fig

def target_chart(df):
    counts = df["Target"].dropna().astype(int).value_counts().sort_index().rename(index={0:"Non-positive",1:"Positive"}).reset_index()
    counts.columns = ["Class", "Rows"]
    return px.bar(counts, x="Class", y="Rows", title="Five-session target distribution", text="Rows")
