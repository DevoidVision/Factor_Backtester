import pandas as pd
import numpy as np

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