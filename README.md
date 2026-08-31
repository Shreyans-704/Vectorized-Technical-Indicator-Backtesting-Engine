````markdown
# Vectorized Technical Indicator Backtesting Engine

A from-scratch, fully vectorized Python backtesting engine for evaluating technical-indicator trading strategies without TA-Lib or third-party backtesting frameworks.

The engine implements technical indicators, signal generation, position management, transaction costs, parameter sweeps, performance metrics, and visualization using NumPy and Pandas.

---

## Overview

Backtesting a trading strategy involves simulating how a strategy would have performed historically while accounting for trading signals, positions, returns, transaction costs, and risk.

Many simple backtesting implementations use row-by-row Python loops:

```python
for i in range(len(data)):
    # calculate signal
    # update position
    # calculate P&L
````

This approach becomes inefficient when testing many parameter combinations or multiple securities.

This project instead uses **vectorized NumPy/Pandas operations** throughout the core calculation pipeline.

```text
Market Data
     │
     ▼
Technical Indicators
     │
     ▼
Trading Signals
     │
     ▼
Lagged Positions
     │
     ▼
Returns + Transaction Costs
     │
     ▼
Performance Metrics
     │
     ▼
Visualization / Parameter Analysis
```

The goal is to provide a fast, transparent, and auditable foundation for quantitative strategy research.

---

## Key Features

* 8 technical indicators implemented from scratch
* Fully vectorized indicator and P&L calculations
* Explicit lookahead-bias protection
* Crossover and threshold-based trading signals
* Long/short position handling
* Configurable transaction costs
* Parameter sweeps
* Sharpe, Sortino, Calmar and drawdown analysis
* Alpha and beta calculation
* Equity curve and drawdown visualization
* Parameter-performance heatmaps
* Historical market-data caching
* Pytest-based automated tests
* Interactive Jupyter notebook for experimentation

---

## Technical Indicators

The engine implements the following indicators without TA-Lib:

| Indicator       | Description                                         |
| --------------- | --------------------------------------------------- |
| SMA             | Simple Moving Average                               |
| EMA             | Exponential Moving Average                          |
| RSI             | Relative Strength Index                             |
| MACD            | Moving Average Convergence Divergence               |
| Bollinger Bands | Volatility bands around a moving average            |
| ATR             | Average True Range                                  |
| Stochastic      | Momentum oscillator based on the recent price range |
| OBV             | On-Balance Volume                                   |

---

## Architecture

```text
┌─────────────────────────────────────────────┐
│                  Data Layer                 │
│        yFinance + Local CSV Cache           │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              Indicator Layer               │
│                                             │
│ SMA │ EMA │ RSI │ MACD │ BBands │ ATR      │
│ Stochastic │ OBV                            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                Signal Layer                │
│                                             │
│ Crossover / Threshold Logic                │
│ Signal ∈ {-1, 0, +1}                       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             Backtest Engine                │
│                                             │
│ Lagged Positions                            │
│ Vectorized Returns                          │
│ Transaction Costs                           │
│ Parameter Sweeps                            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             Metrics Layer                  │
│                                             │
│ Sharpe │ Sortino │ Calmar │ Drawdown       │
│ Alpha │ Beta │ Return │ Volatility         │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│           Visualization Layer              │
│                                             │
│ Equity Curves │ Drawdowns │ Heatmaps       │
└─────────────────────────────────────────────┘
```

---

## Project Structure

```text
quant-backtest-engine/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── signals.py
│   ├── backtester.py
│   ├── metrics.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   └── test_indicators.py
│
├── notebooks/
│   └── Technical_Indicator_Backtesting_Engine.ipynb
│
├── data/
│   └── cache/
│
└── reports/
    └── .gitkeep
```

### Module Responsibilities

| Module               | Purpose                                               |
| -------------------- | ----------------------------------------------------- |
| `data_loader.py`     | Downloads, validates and caches historical OHLCV data |
| `indicators.py`      | Implements technical indicators from scratch          |
| `signals.py`         | Converts indicator values into trading signals        |
| `backtester.py`      | Runs vectorized backtests and parameter sweeps        |
| `metrics.py`         | Calculates performance and risk statistics            |
| `visualization.py`   | Generates performance charts and heatmaps             |
| `test_indicators.py` | Tests indicator calculations and expected properties  |

---

## Lookahead-Bias Protection

A major concern in historical backtesting is **lookahead bias** — using information that would not have been available when a trading decision was made.

The engine explicitly separates signal generation from trade execution.

Signals are shifted by one period:

```python
positions = signal.shift(1)
```

Therefore:

```text
position[t] = signal[t-1]
```

A signal generated using information from period `t` can only affect the position from period `t+1`.

This makes the execution timing explicit and easier to audit.

---

## Vectorized P&L Calculation

The core return calculation uses vectorized operations rather than iterating through individual rows.

Conceptually:

```python
market_returns = close.pct_change()

positions = signal.shift(1)

strategy_returns = positions * market_returns
```

Transaction costs are incorporated based on changes in position:

```python
turnover = positions.diff().abs()

costs = turnover * transaction_cost

net_returns = strategy_returns - costs
```

This approach allows the engine to process complete time series using optimized NumPy/Pandas operations.

---

## Example Strategy

A simple moving-average crossover strategy can be represented as:

```text
Fast SMA > Slow SMA
        │
        ▼
     Long (+1)

Fast SMA < Slow SMA
        │
        ▼
     Short (-1)
```

Example:

```python
fast_sma = sma(close, 20)
slow_sma = sma(close, 50)

signal = np.where(
    fast_sma > slow_sma,
    1,
    -1
)

positions = pd.Series(signal, index=close.index).shift(1)
```

The one-period shift prevents the strategy from trading on the same observation used to generate the signal.

---

## Parameter Sweeps

The vectorized architecture makes it possible to systematically evaluate different strategy parameters.

For example:

```text
Fast SMA: 5, 10, 20, 30
Slow SMA: 30, 50, 100, 200
```

The resulting parameter combinations can be evaluated using performance and risk metrics.

Example workflow:

```python
results = run_parameter_sweep(
    ticker="AAPL",
    fast_periods=[5, 10, 20, 30],
    slow_periods=[30, 50, 100, 200]
)
```

Results can then be compared using tables and heatmaps.

---

## Performance Metrics

The metrics layer provides several measures for evaluating strategy performance.

### Sharpe Ratio

Measures risk-adjusted return relative to return volatility.

### Sortino Ratio

Measures risk-adjusted return while focusing on downside volatility.

### Maximum Drawdown

Measures the largest peak-to-trough decline in portfolio value.

```text
Drawdown = Equity Curve / Running Maximum - 1
```

### Calmar Ratio

Compares annualized return against maximum drawdown.

### Alpha and Beta

Measures strategy performance relative to a benchmark.

Additional metrics include total return and annualized volatility.

---

## Data Layer

Historical OHLCV data is obtained through Yahoo Finance.

The data layer supports local caching to avoid repeatedly downloading the same historical dataset.

Typical fields include:

```text
Date
Open
High
Low
Close
Volume
```

Downloaded cache files are excluded from version control.

---

## Quickstart

### Clone the Repository

```bash
git clone https://github.com/Shreyans-704/Vectorized-Technical-Indicator-Backtesting-Engine.git

cd Vectorized-Technical-Indicator-Backtesting-Engine
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/ -v
```

### Run the Notebook

Open:

```text
notebooks/Technical_Indicator_Backtesting_Engine.ipynb
```

Run the notebook from top to bottom to execute the complete pipeline:

```text
Data → Indicators → Signals → Backtest → Metrics → Visualization
```

---

## Google Colab

The included Jupyter notebook can also be run in Google Colab.

The configuration section allows you to change the ticker and historical period:

```python
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
```

This makes it easy to experiment with different securities and time periods without modifying the core engine.

---

## Testing

The project uses `pytest` for automated testing.

Run:

```bash
pytest tests/ -v
```

Tests validate the indicator implementations and expected mathematical properties, including:

* Output length
* Warm-up period behavior
* Valid value ranges
* Handling of controlled input data
* Consistency with indicator definitions

---

## Design Principles

### Vectorization First

Use NumPy/Pandas operations instead of explicit Python loops wherever practical.

### Explicit Execution Timing

Separate signal generation from position execution to make trading timing clear.

### Reproducibility

Keep data loading, indicators, signals, backtesting and metrics in separate modules.

### Auditability

Make every stage of the backtesting pipeline independently inspectable.

### No Black-Box Backtesting Framework

Core indicators and backtesting logic are implemented from scratch rather than delegated to a specialized trading framework.

---

## Limitations

This project is intended as a research and engineering tool rather than a production trading system.

Important limitations include:

* Historical performance does not guarantee future performance.
* Slippage and market impact are simplified.
* Transaction costs are based on configurable assumptions.
* Order-book dynamics are not modeled.
* Data-source limitations can introduce survivorship or corporate-action issues.
* Parameter optimization can result in overfitting.
* Out-of-sample validation is necessary before drawing conclusions about strategy robustness.

A strategy that performs well in a historical backtest is not necessarily profitable in live markets.

---

## Tech Stack

* **Python**
* **NumPy**
* **Pandas**
* **yFinance**
* **Matplotlib**
* **Seaborn**
* **Pytest**
* **Jupyter Notebook**
* **Git / GitHub**

---

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

## Disclaimer

This project is intended for educational, research, and software-engineering purposes only.

Nothing in this repository constitutes financial advice or a recommendation to buy or sell any security.

```
```
