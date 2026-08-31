"""
test_indicators.py
===================
Pytest suite validating the vectorized indicator library against
known properties (not just "does it run") -- e.g. RSI must stay in
[0, 100], Bollinger middle band must equal the SMA, MACD histogram
must equal macd - signal, etc.
"""

import numpy as np
import pandas as pd
import pytest

from src import indicators as ind


@pytest.fixture
def price_series() -> pd.Series:
    """Deterministic synthetic price series (geometric random walk)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = rng.normal(loc=0.0003, scale=0.012, size=len(dates))
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def ohlcv(price_series) -> pd.DataFrame:
    """Synthetic OHLCV built around the close series, with plausible high/low/volume."""
    rng = np.random.default_rng(7)
    close = price_series
    high = close * (1 + rng.uniform(0, 0.01, size=len(close)))
    low = close * (1 - rng.uniform(0, 0.01, size=len(close)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, size=len(close)), index=close.index)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


# --- SMA -----------------------------------------------------------------

def test_sma_matches_manual_mean(price_series):
    result = ind.sma(price_series, window=10)
    manual = price_series.iloc[0:10].mean()
    assert result.iloc[9] == pytest.approx(manual)
    assert result.iloc[:9].isna().all()  # warm-up period must be NaN, not a partial average


def test_sma_rejects_bad_window(price_series):
    with pytest.raises(ValueError):
        ind.sma(price_series, window=0)


# --- EMA -----------------------------------------------------------------

def test_ema_converges_toward_price_in_flat_series():
    flat = pd.Series([50.0] * 100)
    result = ind.ema(flat, span=10)
    assert result.iloc[-1] == pytest.approx(50.0, abs=1e-6)


# --- RSI -----------------------------------------------------------------

def test_rsi_bounded_between_0_and_100(price_series):
    result = ind.rsi(price_series, window=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_is_100_for_pure_uptrend():
    uptrend = pd.Series(np.arange(1, 30, dtype=float))
    result = ind.rsi(uptrend, window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


# --- MACD ------------------------------------------------------------------

def test_macd_histogram_equals_macd_minus_signal(price_series):
    result = ind.macd(price_series)
    pd.testing.assert_series_equal(
        result["histogram"], (result["macd"] - result["signal"]).rename("histogram")
    )


def test_macd_rejects_fast_greater_than_slow(price_series):
    with pytest.raises(ValueError):
        ind.macd(price_series, fast=30, slow=12)


# --- Bollinger Bands -------------------------------------------------------

def test_bollinger_middle_band_equals_sma(price_series):
    bands = ind.bollinger_bands(price_series, window=20)
    sma20 = ind.sma(price_series, window=20)
    pd.testing.assert_series_equal(bands["middle"], sma20, check_names=False)


def test_bollinger_upper_always_above_lower(price_series):
    bands = ind.bollinger_bands(price_series, window=20).dropna()
    assert (bands["upper"] >= bands["lower"]).all()


# --- ATR -------------------------------------------------------------------

def test_atr_is_non_negative(ohlcv):
    result = ind.atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], window=14)
    assert (result.dropna() >= 0).all()


# --- Stochastic Oscillator --------------------------------------------------

def test_stochastic_percent_k_bounded(ohlcv):
    result = ind.stochastic_oscillator(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
    valid = result["percent_k"].dropna()
    assert (valid >= -1e-9).all() and (valid <= 100 + 1e-9).all()


# --- OBV -------------------------------------------------------------------

def test_obv_increases_on_up_day_decreases_on_down_day():
    close = pd.Series([10, 11, 10.5])
    volume = pd.Series([1000, 2000, 1500])
    result = ind.obv(close, volume)
    assert result.iloc[1] == result.iloc[0] + 2000  # price went up -> add volume
    assert result.iloc[2] == result.iloc[1] - 1500  # price went down -> subtract volume


def test_obv_requires_matching_index():
    close = pd.Series([10, 11], index=[0, 1])
    volume = pd.Series([1000, 2000], index=[1, 2])
    with pytest.raises(ValueError):
        ind.obv(close, volume)
