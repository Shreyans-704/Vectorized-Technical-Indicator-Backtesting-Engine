"""
signals.py
==========
Converts raw indicator output into discrete trading signals.

Signal convention used throughout this engine:
    +1  -> hold/enter a long position
    -1  -> hold/enter a short position (or "flat" if shorting disabled)
     0  -> no information yet (warm-up period) / explicit hold

All functions are vectorized: they operate on entire Series at once
using NumPy boolean masks, never a per-row Python loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover_signal(fast_sma: pd.Series, slow_sma: pd.Series) -> pd.Series:
    """
    Trend-following signal: long when the fast SMA is above the slow SMA
    (short-term trend stronger than long-term trend), short otherwise.
    """
    if not fast_sma.index.equals(slow_sma.index):
        raise ValueError("fast_sma and slow_sma must share the same index")

    signal = pd.Series(np.where(fast_sma > slow_sma, 1, -1), index=fast_sma.index)
    warm_up = fast_sma.isna() | slow_sma.isna()
    signal[warm_up] = 0
    return signal.rename("signal_sma_crossover")


def rsi_threshold_signal(rsi_series: pd.Series, lower: float = 30, upper: float = 70) -> pd.Series:
    """
    Mean-reversion signal: buy when RSI signals "oversold" (< lower),
    sell when RSI signals "overbought" (> upper), otherwise hold
    whatever position was already open (no opinion in the neutral zone).
    """
    if lower >= upper:
        raise ValueError(f"lower threshold ({lower}) must be < upper threshold ({upper})")

    signal = pd.Series(0, index=rsi_series.index, dtype=float)
    signal[rsi_series < lower] = 1
    signal[rsi_series > upper] = -1
    signal[rsi_series.isna()] = 0
    # Carry the last non-neutral signal forward through the "no opinion" zone.
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    return signal.rename("signal_rsi_threshold")


def macd_crossover_signal(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """Long when the MACD line is above its signal line, short otherwise."""
    if not macd_line.index.equals(signal_line.index):
        raise ValueError("macd_line and signal_line must share the same index")

    signal = pd.Series(np.where(macd_line > signal_line, 1, -1), index=macd_line.index)
    warm_up = macd_line.isna() | signal_line.isna()
    signal[warm_up] = 0
    return signal.rename("signal_macd_crossover")


def bollinger_band_signal(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """
    Mean-reversion signal: buy when price touches/pierces the lower band
    (perceived oversold), sell when it touches/pierces the upper band.
    Holds the prior signal in between bands.
    """
    if not (close.index.equals(lower.index) and close.index.equals(upper.index)):
        raise ValueError("close, lower, and upper must share the same index")

    signal = pd.Series(0, index=close.index, dtype=float)
    signal[close <= lower] = 1
    signal[close >= upper] = -1
    signal[lower.isna() | upper.isna()] = 0
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    return signal.rename("signal_bollinger")


def stochastic_threshold_signal(percent_k: pd.Series, lower: float = 20, upper: float = 80) -> pd.Series:
    """Buy when %K < lower (oversold), sell when %K > upper (overbought)."""
    signal = pd.Series(0, index=percent_k.index, dtype=float)
    signal[percent_k < lower] = 1
    signal[percent_k > upper] = -1
    signal[percent_k.isna()] = 0
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    return signal.rename("signal_stochastic")
