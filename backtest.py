import pandas as pd
import numpy as np

from visuals import plot_cumulative_returns, plot_drawdown, plot_factor_heatmap

def monthly_rebalance(dates, scores, prices, top_n=10, weighting='equal', optimizer=None):
    """
    Simulate monthly rebalancing: select top_n stocks by score, assign weights, and track holdings.
    Args:
        dates (list): List of rebalance dates.
        scores (pd.DataFrame): DataFrame of scores (date x ticker).
        prices (pd.DataFrame): Price DataFrame (date x ticker).
        top_n (int): Number of stocks to select.
        weighting (str): 'equal' or 'opt' (PyPortfolioOpt).
        optimizer (callable): Optional optimizer function.
    Returns:
        pd.DataFrame: Portfolio weights over time (date x ticker).
    """
    weights = pd.DataFrame(index=dates, columns=prices.columns).fillna(0)
    for date in dates:
        top = scores.loc[date].nlargest(top_n).index
        if weighting == 'equal' or optimizer is None:
            w = pd.Series(1/top_n, index=top)
        else:
            w = optimizer(prices.loc[:date, top])
        weights.loc[date, top] = w
    return weights

def simulate_portfolio(prices, weights):
    """
    Simulate portfolio returns given price and weight DataFrames.
    Args:
        prices (pd.DataFrame): Price DataFrame (date x ticker).
        weights (pd.DataFrame): Portfolio weights (date x ticker).
    Returns:
        pd.Series: Portfolio value over time.
    """
    returns = prices.pct_change().fillna(0)
    port_rets = (weights.shift().fillna(0) * returns).sum(axis=1)
    port_value = (1 + port_rets).cumprod()
    return port_value

def calculate_sharpe(returns, risk_free=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio.
    Args:
        returns (pd.Series): Daily returns.
        risk_free (float): Risk-free rate.
        periods_per_year (int): Number of periods per year.
    Returns:
        float: Sharpe ratio.
    """
    excess = returns - risk_free/periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def calculate_max_drawdown(values):
    """
    Calculate max drawdown from portfolio value series.
    Args:
        values (pd.Series): Portfolio value over time.
    Returns:
        float: Max drawdown (as positive number).
    """
    roll_max = values.cummax()
    drawdown = (values - roll_max) / roll_max
    return drawdown.min()

def calculate_cagr(values, periods_per_year=252):
    """
    Calculate CAGR from portfolio value series.
    Args:
        values (pd.Series): Portfolio value over time.
        periods_per_year (int): Number of periods per year.
    Returns:
        float: CAGR.
    """
    n = len(values)
    return (values.iloc[-1] / values.iloc[0]) ** (periods_per_year / n) - 1 
# ...existing code...

def run_backtest(start, end, factors, top_n):
    """
    High-level wrapper to run full backtest and return plots.
    """
    from factors import get_stock_universe, calculate_factors
    import yfinance as yf

    # 1. Get list of stock tickers
    tickers = get_stock_universe()

    # 2. Download data
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data = data.dropna(axis=1)
    data = data.fillna(method='ffill').fillna(method='bfill')

    # 3. Calculate factor scores (should return DataFrame: date x ticker)
    factor_scores = calculate_factors(data, factors)

    # 4. Monthly rebalance dates
    rebalance_dates = data.resample('M').last().index

    # 5. Generate weights using your function
    weights = monthly_rebalance(rebalance_dates, factor_scores, data, top_n=top_n)
    weights = weights.reindex(data.index, method='ffill').fillna(0)

    # 6. Simulate portfolio
    portfolio_value = simulate_portfolio(data, weights)

    # 7. Generate plots
    cumulative_plot = plot_cumulative_returns(portfolio_value)
    drawdown_plot = plot_drawdown(portfolio_value)
    heatmap = plot_factor_heatmap(factor_scores)

    return {
        "cumulative_plot": cumulative_plot,
        "drawdown_plot": drawdown_plot,
        "heatmap": heatmap
    }
