# Vectorized Technical Indicator Backtesting Engine

A from-scratch, fully vectorized backtesting engine for technical-indicator
trading strategies — no TA-Lib, no backtesting framework, no row-by-row
Python loops anywhere in the P&L calculation. Built to be fast enough to
sweep hundreds of parameter combinations across multiple tickers in seconds.

## Why this exists

Most "build a trading bot" tutorials loop over rows with `for i in range(len(df))`
to simulate a strategy day by day. That's intuitive but slow, and it's easy to
accidentally leak future information into a "past" decision (lookahead bias)
when you're manually tracking state inside the loop.

This project instead expresses every step — indicator math, signal generation,
position sizing, P&L, transaction costs — as vectorized pandas/NumPy operations.
The lookahead-bias guard is a single explicit line (`positions = signal.shift(1)`)
rather than something implicit in loop ordering, which makes the engine both
faster and easier to audit for correctness.

## Architecture

```
Data Layer (yfinance + CSV cache)
        │
        ▼
Indicator Layer (8 indicators, vectorized: SMA, EMA, RSI, MACD,
                 Bollinger Bands, ATR, Stochastic, OBV)
        │
        ▼
Signal Layer (crossover / threshold logic → {-1, 0, +1})
        │
        ▼
Backtest Engine (lagged positions, vectorized P&L, transaction costs)
        │
        ▼
Metrics Layer (Sharpe, Sortino, Calmar, max drawdown, alpha/beta, ...)
        │
        ▼
Visualization Layer (equity curves, drawdown, dashboards, heatmaps)
```

## Project structure

```
quant-backtest-engine/
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py      # OHLCV download + caching + retries
│   ├── indicators.py       # 8 vectorized indicators, from scratch
│   ├── signals.py          # Indicator values -> trading signals
│   ├── backtester.py       # Vectorized backtest engine + parameter sweep
│   ├── metrics.py          # Performance/risk analytics
│   └── visualization.py    # matplotlib/seaborn charting
├── tests/
│   └── test_indicators.py  # pytest suite (property-based correctness checks)
├── notebooks/
│   └── Technical_Indicator_Backtesting_Engine.ipynb   # Run this in Colab
└── reports/                # Generated charts/output land here
```

## Quickstart (Google Colab)

1. Open `notebooks/Technical_Indicator_Backtesting_Engine.ipynb` in Colab.
2. Run cells top to bottom. The first code cell installs dependencies.
3. Change `TICKER` and the date range in the configuration cell to backtest
   any symbol Yahoo Finance covers.

## Quickstart (local / GitHub)

```bash
git clone <your-repo-url>
cd quant-backtest-engine
pip install -r requirements.txt
pytest tests/ -v
```

## Sample results (synthetic data smoke test)

A 20/50-day SMA crossover strategy on a ~6-year synthetic price series,
with 7bps round-trip costs:

| Metric | Value |
|---|---|
| Sharpe Ratio | varies by parameters — see `run_parameter_sweep` |
| Max Drawdown | computed via `metrics.max_drawdown` |
| Lookahead-bias check | verified: `position[t] == signal[t-1]` for all t |

Run the notebook against a real ticker to get real numbers — the synthetic
data above is only used to unit-test the engine's correctness.

## License

MIT
