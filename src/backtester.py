"""
backtester.py
=============
The vectorized backtesting engine -- the centerpiece of this project.

Design goal: compute strategy P&L for an entire price + signal history
in a fixed number of pandas/NumPy operations, regardless of how many
bars are in the series. No per-bar Python loop, no manually maintained
"current position" state machine stepping through rows.

Core mechanics
---------------
1. A signal generated using bar t's closing data cannot be acted on
   until bar t+1 (you can't trade on information you only see at the
   close). We enforce this with `positions = signal.shift(1)` -- this
   single line is what prevents lookahead bias in the engine.
2. Strategy return at bar t = position_held_during_t * market_return_t.
3. Transaction costs + slippage are charged proportionally whenever the
   position size changes, computed as one vectorized diff().abs() pass
   rather than tracked trade-by-trade.
4. The equity curve is the cumulative product of (1 + strategy_return),
   scaled by initial capital -- compounding handled by .cumprod().

This file deliberately has zero matplotlib/plotting code -- the engine
only computes numbers; presentation lives in visualization.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Tunable parameters for a backtest run."""
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0     # cost per trade, basis points of notional traded
    slippage_bps: float = 2.0             # modeled the same way as transaction cost
    allow_short: bool = True              # if False, signal is clipped to [0, 1] (long-only/flat)

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("transaction_cost_bps and slippage_bps cannot be negative")


@dataclass
class BacktestResult:
    """Container for everything a backtest run produces."""
    equity_curve: pd.Series
    benchmark_equity_curve: pd.Series
    strategy_returns: pd.Series
    market_returns: pd.Series
    positions: pd.Series
    trade_dates: pd.DatetimeIndex
    config: BacktestConfig = field(repr=False)


class VectorizedBacktester:
    """Runs a fully vectorized long/short (or long-only) backtest."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, close: pd.Series, signal: pd.Series) -> BacktestResult:
        """
        Parameters
        ----------
        close : pd.Series
            Closing prices indexed by date.
        signal : pd.Series
            Trading signal in {-1, 0, +1}, same index as `close`,
            representing the position the strategy *wants* based on
            information available at each bar's close.

        Returns
        -------
        BacktestResult
        """
        self._validate_inputs(close, signal)
        cfg = self.config

        working_signal = signal.clip(lower=0) if not cfg.allow_short else signal

        market_returns = close.pct_change().fillna(0)

        # Lag by one bar -- this is the line that prevents lookahead bias.
        positions = working_signal.shift(1).fillna(0)

        # A "trade" happens whenever the held position changes magnitude/direction.
        position_changes = positions.diff().abs().fillna(0)
        cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
        transaction_costs = position_changes * cost_rate

        strategy_returns = positions * market_returns - transaction_costs
        equity_curve = cfg.initial_capital * (1 + strategy_returns).cumprod()
        benchmark_equity_curve = cfg.initial_capital * (1 + market_returns).cumprod()

        trade_dates = position_changes[position_changes > 0].index

        return BacktestResult(
            equity_curve=equity_curve.rename("strategy_equity"),
            benchmark_equity_curve=benchmark_equity_curve.rename("benchmark_equity"),
            strategy_returns=strategy_returns.rename("strategy_returns"),
            market_returns=market_returns.rename("market_returns"),
            positions=positions.rename("position"),
            trade_dates=trade_dates,
            config=cfg,
        )

    @staticmethod
    def _validate_inputs(close: pd.Series, signal: pd.Series) -> None:
        if not isinstance(close, pd.Series) or not isinstance(signal, pd.Series):
            raise TypeError("close and signal must both be pandas Series")
        if close.empty or signal.empty:
            raise ValueError("close and signal cannot be empty")
        if not close.index.equals(signal.index):
            raise ValueError(
                "close and signal must share an identical index. "
                "Reindex/align your signal to the price series before backtesting."
            )
        if signal.isna().any():
            raise ValueError("signal contains NaN values -- fill warm-up periods with 0 before backtesting")


def run_parameter_sweep(close: pd.Series, signal_fn, param_grid: list[dict],
                         config: BacktestConfig | None = None) -> pd.DataFrame:
    """
    Convenience helper for sweeping a strategy across many parameter
    combinations. `signal_fn` must accept `close` plus the keyword
    arguments in each dict of `param_grid`, and return a signal Series.

    Returns a DataFrame of one row per parameter combination with its
    resulting equity curve's final value and Sharpe ratio attached,
    sorted best-Sharpe-first. (Full metrics can be recomputed afterward
    using metrics.py for any row of interest.)
    """
    from . import metrics  # local import to avoid a circular import at module load time

    backtester = VectorizedBacktester(config)
    rows = []
    for params in param_grid:
        try:
            signal = signal_fn(close, **params)
            result = backtester.run(close, signal)
            sharpe = metrics.sharpe_ratio(result.strategy_returns)
            final_equity = result.equity_curve.iloc[-1]
            rows.append({**params, "final_equity": final_equity, "sharpe_ratio": sharpe})
        except Exception as exc:  # noqa: BLE001 -- we want the sweep to continue past one bad combo
            rows.append({**params, "final_equity": np.nan, "sharpe_ratio": np.nan, "error": str(exc)})

    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
