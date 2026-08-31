"""
indicators.py
=============
Vectorized technical indicator library, built from scratch on top of
NumPy and pandas only (no TA-Lib, no `ta` package, no third-party
indicator dependency).

Why "vectorized" matters
-------------------------
A naive implementation loops over each row in Python to compute a
moving average or RSI value. For a 10-year daily series (~2,500 rows)
that's slow but tolerable. The moment you want to backtest 8
indicators x 50 parameter combinations x 20 tickers, a row-by-row
Python loop becomes the bottleneck (tens of thousands of slow loops).

Every function below instead expresses the calculation as a single
pandas/NumPy operation (rolling window, exponential weighting, or
array broadcasting) that is executed in optimized C under the hood.
There is no `for i in range(len(series))` anywhere in this file.

Conventions
-----------
- All functions accept and return pandas Series/DataFrame objects
  indexed by datetime, aligned to the input index.
- Leading rows where a full lookback window isn't yet available are
  returned as NaN rather than silently using a partial window -- this
  prevents subtly wrong values from leaking into a backtest.
- Functions raise informative errors on bad input rather than failing
  with a cryptic pandas traceback three layers down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Input validation helpers
# ----------------------------------------------------------------------------

def _validate_series(series: pd.Series, name: str = "series") -> None:
    """Raise a clear error if `series` isn't a usable pandas Series."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series).__name__}")
    if series.empty:
        raise ValueError(f"{name} is empty")
    if series.isna().all():
        raise ValueError(f"{name} contains only NaN values")


def _validate_window(window: int, name: str = "window") -> None:
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window}")


# ----------------------------------------------------------------------------
# 1. Simple Moving Average
# ----------------------------------------------------------------------------

def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average: the unweighted mean of the last `window` prices.

    Vectorization: pandas .rolling().mean() computes every window's mean
    via a single optimized pass (sliding-sum algorithm), not a Python loop.
    """
    _validate_series(series, "series")
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).mean().rename(f"SMA_{window}")


# ----------------------------------------------------------------------------
# 2. Exponential Moving Average
# ----------------------------------------------------------------------------

def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average: weights recent prices more heavily,
    with weight decaying exponentially the further back in time.

    Vectorization: pandas .ewm().mean() applies the recursive EMA
    formula using a compiled C routine internally -- the recursion
    is hidden from the Python layer, so from our side it's one call.
    """
    _validate_series(series, "series")
    _validate_window(span, "span")
    return series.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"EMA_{span}")


# ----------------------------------------------------------------------------
# 3. Relative Strength Index (Wilder's RSI)
# ----------------------------------------------------------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index, using Wilder's original smoothing
    (an EMA with alpha = 1/window, which is the textbook RSI formula,
    not a plain SMA-based approximation).

    RSI = 100 - 100 / (1 + RS),  RS = avg_gain / avg_loss
    """
    _validate_series(series, "series")
    _validate_window(window)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    # When avg_loss is exactly 0 (pure uptrend), RSI is defined as 100.
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values.rename(f"RSI_{window}")


# ----------------------------------------------------------------------------
# 4. MACD (Moving Average Convergence Divergence)
# ----------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD line, span=signal)
    Histogram = MACD line - Signal line

    Returns a DataFrame with columns: macd, signal, histogram.
    """
    _validate_series(series, "series")
    if fast >= slow:
        raise ValueError(f"fast period ({fast}) must be less than slow period ({slow})")
    _validate_window(signal, "signal")

    ema_fast = ema(series, span=fast)
    ema_slow = ema(series, span=slow)
    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean().rename("signal")
    histogram = (macd_line - signal_line).rename("histogram")

    return pd.concat([macd_line, signal_line, histogram], axis=1)


# ----------------------------------------------------------------------------
# 5. Bollinger Bands
# ----------------------------------------------------------------------------

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: an SMA "middle band" with upper/lower bands placed
    `num_std` standard deviations away, capturing a volatility envelope.

    Also returns %B = (price - lower) / (upper - lower), a normalized
    measure of where price sits within the bands (useful as a signal
    feature: %B > 1 means price pierced the upper band, < 0 the lower).
    """
    _validate_series(series, "series")
    _validate_window(window)
    if num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std}")

    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_width = upper - lower
    percent_b = (series - lower) / band_width.replace(0, np.nan)

    return pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "percent_b": percent_b,
    })


# ----------------------------------------------------------------------------
# 6. Average True Range (volatility)
# ----------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range: Wilder's volatility measure. True Range accounts
    for gaps between sessions by taking the largest of three measures:
    high-low, |high - prev_close|, |low - prev_close|.
    """
    for s, name in ((high, "high"), (low, "low"), (close, "close")):
        _validate_series(s, name)
    _validate_window(window)

    prev_close = close.shift(1)
    tr_components = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    true_range = tr_components.max(axis=1)

    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().rename(f"ATR_{window}")


# ----------------------------------------------------------------------------
# 7. Stochastic Oscillator
# ----------------------------------------------------------------------------

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                           k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_window)  -- a smoothed signal line for %K.
    """
    for s, name in ((high, "high"), (low, "low"), (close, "close")):
        _validate_series(s, name)
    _validate_window(k_window, "k_window")
    _validate_window(d_window, "d_window")

    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)

    percent_k = 100 * (close - lowest_low) / denom
    percent_d = percent_k.rolling(window=d_window, min_periods=d_window).mean()

    return pd.DataFrame({"percent_k": percent_k, "percent_d": percent_d})


# ----------------------------------------------------------------------------
# 8. On-Balance Volume
# ----------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: a running cumulative total of volume, added on up
    days and subtracted on down days. Used as a volume-confirmation
    indicator for price trends.
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")
    if not close.index.equals(volume.index):
        raise ValueError("close and volume must share the same index")

    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("OBV")


# ----------------------------------------------------------------------------
# Registry: lets the rest of the engine iterate over "all indicators"
# generically (used by the notebook's comparison/dashboard cells).
# ----------------------------------------------------------------------------

INDICATOR_REGISTRY = {
    "SMA": sma,
    "EMA": ema,
    "RSI": rsi,
    "MACD": macd,
    "BollingerBands": bollinger_bands,
    "ATR": atr,
    "Stochastic": stochastic_oscillator,
    "OBV": obv,
}
