"""
visualization.py
=================
Professional, presentation-quality charts for the backtesting engine,
built on matplotlib/seaborn. Every function returns the Figure object
(rather than calling plt.show() internally) so notebooks can display,
save, or compose them as needed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="deep")

COLOR_STRATEGY = "#1f77b4"
COLOR_BENCHMARK = "#7f7f7f"
COLOR_DRAWDOWN = "#d62728"
COLOR_LONG = "#2ca02c"
COLOR_SHORT = "#d62728"


def plot_price_with_indicator(close: pd.Series, overlays: dict[str, pd.Series],
                               title: str = "Price with Indicator Overlay") -> plt.Figure:
    """Plots close price with one or more indicator lines overlaid (e.g. SMA20 vs SMA50)."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(close.index, close.values, label="Close", color="black", linewidth=1.1)
    for label, series in overlays.items():
        ax.plot(series.index, series.values, label=label, linewidth=1.4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    return fig


def plot_equity_curve(equity_curve: pd.Series, benchmark_equity_curve: pd.Series,
                       title: str = "Strategy vs. Benchmark Equity Curve") -> plt.Figure:
    """Compares a strategy's growth-of-$X to a buy-and-hold benchmark on the same axis."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", color=COLOR_STRATEGY, linewidth=1.8)
    ax.plot(benchmark_equity_curve.index, benchmark_equity_curve.values, label="Buy & Hold",
            color=COLOR_BENCHMARK, linewidth=1.4, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_drawdown(drawdown: pd.Series, title: str = "Strategy Drawdown") -> plt.Figure:
    """Underwater plot: how far below the running peak the strategy is at every point in time."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color=COLOR_DRAWDOWN, alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color=COLOR_DRAWDOWN, linewidth=1.0)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    return fig


def plot_returns_distribution(returns: pd.Series, title: str = "Strategy Daily Returns Distribution") -> plt.Figure:
    """Histogram + KDE of daily returns, useful for eyeballing fat tails / skew."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(returns.dropna() * 100, bins=60, kde=True, color=COLOR_STRATEGY, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 63,
                         title: str = "Rolling Sharpe Ratio (63-day)") -> plt.Figure:
    """Rolling annualized Sharpe -- reveals regime changes a single full-sample Sharpe hides."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLOR_STRATEGY, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    fig.tight_layout()
    return fig


def plot_positions(close: pd.Series, positions: pd.Series,
                    title: str = "Position Over Time") -> plt.Figure:
    """Shades the price chart green while long and red while short, for a quick visual sanity check."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(close.index, close.values, color="black", linewidth=1.0, label="Close")

    long_mask = positions > 0
    short_mask = positions < 0
    ax.fill_between(close.index, close.min(), close.max(), where=long_mask,
                     color=COLOR_LONG, alpha=0.12, label="Long")
    ax.fill_between(close.index, close.min(), close.max(), where=short_mask,
                     color=COLOR_SHORT, alpha=0.12, label="Short")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    return fig


def plot_strategy_comparison(equity_curves: dict[str, pd.Series],
                              title: str = "Strategy Comparison") -> plt.Figure:
    """Overlays multiple strategies' equity curves -- the core 'dashboard' comparison chart."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=label, linewidth=1.6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(metrics_df: pd.DataFrame, title: str = "Strategy Metrics Comparison") -> plt.Figure:
    """
    Heatmap of metrics (rows) x strategies (columns), z-scored per row so
    metrics on very different scales (Sharpe vs. Max Drawdown) are all
    visually comparable on the same color scale.
    """
    z_scored = metrics_df.sub(metrics_df.mean(axis=1), axis=0).div(metrics_df.std(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics_df.columns) + 3, 0.55 * len(metrics_df) + 2))
    sns.heatmap(z_scored, annot=metrics_df.round(3), fmt="", cmap="RdYlGn", center=0,
                cbar_kws={"label": "Relative performance (z-score)"}, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def build_dashboard(close: pd.Series, equity_curve: pd.Series, benchmark_equity_curve: pd.Series,
                     drawdown: pd.Series, returns: pd.Series, positions: pd.Series,
                     title: str = "Backtest Dashboard") -> plt.Figure:
    """
    Single composite figure combining equity curve, drawdown, position
    timeline, and return distribution into one dashboard-style image --
    the kind of summary chart you'd put at the top of a strategy README.
    """
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1.2])

    ax_equity = fig.add_subplot(gs[0, :])
    ax_equity.plot(equity_curve.index, equity_curve.values, label="Strategy", color=COLOR_STRATEGY, linewidth=1.8)
    ax_equity.plot(benchmark_equity_curve.index, benchmark_equity_curve.values, label="Buy & Hold",
                    color=COLOR_BENCHMARK, linewidth=1.3, linestyle="--")
    ax_equity.set_title("Equity Curve: Strategy vs. Benchmark", fontweight="bold")
    ax_equity.legend(loc="upper left")
    ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax_dd = fig.add_subplot(gs[1, :])
    ax_dd.fill_between(drawdown.index, drawdown.values * 100, 0, color=COLOR_DRAWDOWN, alpha=0.4)
    ax_dd.set_title("Drawdown", fontweight="bold")
    ax_dd.set_ylabel("%")

    ax_pos = fig.add_subplot(gs[2, 0])
    long_mask, short_mask = positions > 0, positions < 0
    ax_pos.fill_between(close.index, 0, 1, where=long_mask, color=COLOR_LONG, alpha=0.5, label="Long")
    ax_pos.fill_between(close.index, 0, 1, where=short_mask, color=COLOR_SHORT, alpha=0.5, label="Short")
    ax_pos.set_yticks([])
    ax_pos.set_title("Position Timeline", fontweight="bold")
    ax_pos.legend(loc="upper right", fontsize=8)

    ax_dist = fig.add_subplot(gs[2, 1])
    sns.histplot(returns.dropna() * 100, bins=40, kde=True, color=COLOR_STRATEGY, ax=ax_dist)
    ax_dist.set_title("Daily Return Distribution", fontweight="bold")
    ax_dist.set_xlabel("Daily Return (%)")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
