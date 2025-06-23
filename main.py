import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import os
from factors import calculate_value_factor, calculate_momentum_factor, calculate_volatility_factor, combine_factors
from backtest import monthly_rebalance, simulate_portfolio, calculate_sharpe, calculate_max_drawdown, calculate_cagr
from visuals import plot_cumulative_returns, plot_drawdown, plot_factor_heatmap

# --- Streamlit imports ---
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt

# --- Helper functions ---
def get_sp500_tickers(n=50):
    # For demo: static list of S&P 500 tickers (replace with web scraping if needed)
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'HD', 'MA', 'XOM', 'LLY', 'ABBV', 'MRK', 'PEP',
        'COST', 'AVGO', 'KO', 'TMO', 'MCD', 'WMT', 'CVX', 'BAC', 'ADBE', 'PFE',
        'CSCO', 'ABT', 'ACN', 'DHR', 'DIS', 'LIN', 'VZ', 'WFC', 'INTC', 'TXN',
        'NEE', 'PM', 'UNP', 'MS', 'HON', 'AMGN', 'IBM', 'QCOM', 'LOW', 'SBUX'
    ][:n]

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close']
    pe_ratios = pd.Series({t: yf.Ticker(t).info.get('trailingPE', np.nan) for t in tickers})
    return close, pe_ratios

# --- Main workflow ---
def run_backtest(start, end, factors_list, top_n):
    tickers = get_sp500_tickers(50)
    close, pe_ratios = download_data(tickers, start, end)
    close = close.dropna(axis=1, thresh=int(0.8*len(close)))
    close = close.fillna(method='ffill').fillna(method='bfill')

    # --- Factor calculation ---
    factors = {}
    if 'value' in factors_list:
        factors['value'] = calculate_value_factor(pe_ratios[close.columns])
    if 'momentum' in factors_list:
        factors['momentum'] = calculate_momentum_factor(close)
    if 'volatility' in factors_list:
        factors['volatility'] = -calculate_volatility_factor(close)

    # --- Combine factors ---
    total_score = combine_factors(factors)
    scores_df = pd.DataFrame([total_score], index=[close.index[-1]])
    for date in close.resample('M').last().index:
        sub_close = close.loc[:date]
        sub_factors = {}
        if 'value' in factors_list:
            sub_factors['value'] = calculate_value_factor(pe_ratios[close.columns])
        if 'momentum' in factors_list:
            sub_factors['momentum'] = calculate_momentum_factor(sub_close)
        if 'volatility' in factors_list:
            sub_factors['volatility'] = -calculate_volatility_factor(sub_close)
        scores_df.loc[date] = combine_factors(sub_factors)
    scores_df = scores_df.sort_index()

    # --- Portfolio construction and backtest ---
    rebalance_dates = scores_df.index
    weights = monthly_rebalance(rebalance_dates, scores_df, close, top_n=top_n)
    weights = weights.reindex(close.index, method='ffill').fillna(0)
    port_value = simulate_portfolio(close, weights)
    spy = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)['Close']
    spy = spy.reindex(close.index).fillna(method='ffill')
    spy_value = spy / spy.iloc[0]

    # --- Metrics ---
    port_rets = port_value.pct_change().dropna()
    sharpe = calculate_sharpe(port_rets)
    mdd = calculate_max_drawdown(port_value)
    cagr = calculate_cagr(port_value)

    return port_value, spy_value, scores_df, sharpe, mdd, cagr

def plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def run_streamlit():
    st.title('Factor-Based Portfolio Backtester')
    st.sidebar.header('Backtest Settings')
    start = st.sidebar.date_input('Start Date', pd.to_datetime('2018-01-01'))
    end = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))
    factors_list = st.sidebar.multiselect('Select Factors', ['value', 'momentum', 'volatility'], default=['value', 'momentum', 'volatility'])
    top_n = st.sidebar.slider('Top N Stocks', 5, 20, 10)
    if st.sidebar.button('Run Backtest'):
        with st.spinner('Running backtest...'):
            port_value, spy_value, scores_df, sharpe, mdd, cagr = run_backtest(str(start), str(end), factors_list, top_n)
            st.subheader('Performance Metrics')
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
            st.write(f"**Max Drawdown:** {mdd:.2%}")
            st.write(f"**CAGR:** {cagr:.2%}")

            # Cumulative Returns Plot
            fig1, ax1 = plt.subplots(figsize=(10,6))
            ax1.plot(port_value, label='Portfolio')
            ax1.plot(spy_value, label='Benchmark (SPY)')
            ax1.set_title('Cumulative Returns')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            st.pyplot(fig1)

            # Drawdown Plot
            roll_max = port_value.cummax()
            drawdown = (port_value - roll_max) / roll_max
            fig2, ax2 = plt.subplots(figsize=(10,6))
            ax2.plot(drawdown, color='red')
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown')
            st.pyplot(fig2)

            # Factor Heatmap
            import seaborn as sns
            fig3, ax3 = plt.subplots(figsize=(12,8))
            sns.heatmap(scores_df.T, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Factor Score Heatmap')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Ticker')
            st.pyplot(fig3)

# --- CLI entrypoint ---
def main():
    import sys
    if any('streamlit' in arg for arg in sys.argv):
        # If run with streamlit, launch dashboard
        run_streamlit()
        return
    parser = argparse.ArgumentParser(description='Factor-Based Portfolio Backtester')
    parser.add_argument('--start', type=str, default='2018-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--factors', nargs='+', default=['value', 'momentum', 'volatility'], help='Factors to use')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top stocks to select')
    args = parser.parse_args()

    port_value, spy_value, scores_df, sharpe, mdd, cagr = run_backtest(args.start, args.end, args.factors, args.top_n)
    print(f'Sharpe Ratio: {sharpe:.2f}\nMax Drawdown: {mdd:.2%}\nCAGR: {cagr:.2%}')
    os.makedirs('output', exist_ok=True)
    plot_cumulative_returns(port_value, spy_value)
    plot_drawdown(port_value)
    plot_factor_heatmap(scores_df)

if __name__ == '__main__':
    main() 