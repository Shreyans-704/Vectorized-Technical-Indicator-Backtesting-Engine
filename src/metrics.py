"""
metrics.py
==========
Performance analytics for backtest results -- the same risk/return
statistics used in systematic trading research and PM tear sheets.

Every function takes pandas Series (equity curve and/or returns) and
returns a single float, computed via vectorized NumPy/pandas operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    """Cumulative return over the full backtest period."""
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate, normalizing total return by elapsed years."""
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Standard deviation of returns, annualized by sqrt(trading days)."""
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of total volatility.
    risk_free_rate is an *annual* rate, converted to a daily rate here.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside volatility (negative returns)."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline in the equity curve, as a negative fraction."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Full drawdown time series (for plotting), not just the max."""
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1).rename("drawdown")


def calmar_ratio(equity_curve: pd.Series) -> float:
    """CAGR divided by the magnitude of the max drawdown -- return per unit of pain."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return float(cagr(equity_curve) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero-return bars that were positive."""
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return np.nan
    return float((nonzero > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of winning-bar returns divided by sum of |losing-bar| returns."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity of strategy returns to benchmark returns (linear regression slope)."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    benchmark_var = cov_matrix[1, 1]
    if benchmark_var == 0:
        return np.nan
    return float(cov_matrix[0, 1] / benchmark_var)


def alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Jensen's alpha: excess return not explained by beta-adjusted benchmark exposure."""
    b = beta(strategy_returns, benchmark_returns)
    if np.isnan(b):
        return np.nan
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    strategy_excess = strategy_returns.mean() - daily_rf
    benchmark_excess = benchmark_returns.mean() - daily_rf
    daily_alpha = strategy_excess - b * benchmark_excess
    return float(daily_alpha * TRADING_DAYS_PER_YEAR)


def summary(equity_curve: pd.Series, returns: pd.Series,
            benchmark_returns: pd.Series | None = None, risk_free_rate: float = 0.0) -> pd.Series:
    """One-stop tear-sheet style summary of every metric above."""
    metrics = {
        "Total Return": total_return(equity_curve),
        "CAGR": cagr(equity_curve),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(equity_curve),
        "Calmar Ratio": calmar_ratio(equity_curve),
        "Win Rate": win_rate(returns),
        "Profit Factor": profit_factor(returns),
    }
    if benchmark_returns is not None:
        metrics["Beta"] = beta(returns, benchmark_returns)
        metrics["Alpha (annualized)"] = alpha(returns, benchmark_returns, risk_free_rate)
    return pd.Series(metrics)
