import streamlit as st
from datetime import datetime
from backtest import run_backtest

st.set_page_config(page_title="Factor-Based Portfolio Backtester", layout="wide")
st.title("📊 Factor-Based Portfolio Backtester")

# Sidebar Inputs
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 1, 1))
factors = st.sidebar.multiselect("Select Factors", ["value", "momentum", "volatility"], default=["value", "momentum"])
top_n = st.sidebar.slider("Top N Stocks", min_value=5, max_value=50, value=15)

if st.sidebar.button("Run Backtest"):
    st.info("Running backtest... This may take a minute.")
    results = run_backtest(str(start_date), str(end_date), factors, top_n)

    st.subheader("📈 Cumulative Returns")
    st.pyplot(results["cumulative_plot"]) 
    st.subheader("📉 Drawdown")
    st.pyplot(results["drawdown_plot"])

    st.subheader("🔥 Factor Score Heatmap")
    st.pyplot(results["heatmap"])

    st.success("✅ Backtest completed!")