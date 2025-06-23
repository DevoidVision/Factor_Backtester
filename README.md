# Factor-Based Portfolio Backtester

## Project Overview
A Python-based backtesting system that selects stocks based on financial factors (value, momentum, volatility), simulates portfolio performance, and evaluates key metrics like cumulative return, Sharpe ratio, and max drawdown.

## Features
- Download historical stock data (S&P 500 subset) using yfinance
- Calculate value (P/E), momentum (6/12-month return), and volatility (rolling std) factors
- Rank and select top stocks monthly, with equal or optimized weights
- Simulate monthly rebalancing and track performance vs. benchmark (SPY)
- Compute Sharpe Ratio, Max Drawdown, CAGR
- Visualize cumulative returns, drawdown, and factor heatmap
- Optional Streamlit dashboard for interactive analysis

## Project Structure
- `main.py`: Core logic and CLI
- `factors.py`: Factor calculation functions
- `backtest.py`: Backtesting and rebalancing logic
- `visuals.py`: Plotting and visualization
- `output/`: Example plots and results

## Setup
1. Clone the repo and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Example
```bash
python main.py --start 2018-01-01 --end 2023-01-01 --factors value momentum volatility --top_n 15
```

## Example Output
- Cumulative returns plot (portfolio vs. SPY)
- Drawdown plot
- Factor score heatmap

See the `output/` folder for sample plots.

---

**Note:** For advanced weighting, install `PyPortfolioOpt`. For dashboard, run `streamlit run main.py`. 