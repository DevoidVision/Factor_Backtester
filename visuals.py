import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cumulative_returns(portfolio, benchmark, output_path='output/cumulative_returns.png'):
    """
    Plot cumulative returns of portfolio vs. benchmark.
    """
    plt.figure(figsize=(10,6))
    plt.plot(portfolio, label='Portfolio')
    plt.plot(benchmark, label='Benchmark (SPY)')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_drawdown(portfolio, output_path='output/drawdown.png'):
    """
    Plot drawdown over time.
    """
    roll_max = portfolio.cummax()
    drawdown = (portfolio - roll_max) / roll_max
    plt.figure(figsize=(10,6))
    plt.plot(drawdown, color='red')
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_factor_heatmap(factor_scores, output_path='output/factor_heatmap.png'):
    """
    Plot heatmap of factor scores (date x ticker).
    """
    plt.figure(figsize=(12,8))
    sns.heatmap(factor_scores.T, cmap='coolwarm', center=0)
    plt.title('Factor Score Heatmap')
    plt.xlabel('Date')
    plt.ylabel('Ticker')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close() 