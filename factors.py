import pandas as pd
import numpy as np

def calculate_value_factor(pe_ratios: pd.Series) -> pd.Series:
    """
    Calculate value factor as the inverse of P/E ratio (lower P/E = higher value score).
    Args:
        pe_ratios (pd.Series): Series of P/E ratios indexed by ticker.
    Returns:
        pd.Series: Value scores (higher is better).
    """
    return 1 / pe_ratios.replace(0, np.nan)

def calculate_momentum_factor(prices: pd.DataFrame, window: int = 126) -> pd.Series:
    """
    Calculate momentum factor as past N-day return (default: 6 months ~126 trading days).
    Args:
        prices (pd.DataFrame): Price DataFrame (date x ticker).
        window (int): Lookback window in trading days.
    Returns:
        pd.Series: Momentum scores (higher is better).
    """
    return prices.pct_change(periods=window).iloc[-1]

def calculate_volatility_factor(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Calculate volatility factor as rolling standard deviation of daily returns (lower is better).
    Args:
        prices (pd.DataFrame): Price DataFrame (date x ticker).
        window (int): Rolling window in trading days.
    Returns:
        pd.Series: Volatility scores (lower is better).
    """
    daily_returns = prices.pct_change()
    return daily_returns.rolling(window=window).std().iloc[-1]

def combine_factors(factor_dfs: dict, weights: dict = None) -> pd.Series:
    """
    Combine multiple factor scores into a total score using z-score ranking and optional weights.
    Args:
        factor_dfs (dict): Dict of {factor_name: pd.Series}.
        weights (dict): Dict of {factor_name: weight}. If None, equal weights.
    Returns:
        pd.Series: Combined score (higher is better).
    """
    zscores = {k: (v - v.mean()) / v.std() for k, v in factor_dfs.items()}
    if weights is None:
        weights = {k: 1/len(zscores) for k in zscores}
    total_score = sum(zscores[k] * weights.get(k, 1) for k in zscores)
    return total_score 