"""
data_loader.py
==============
Data collection pipeline: fetches OHLCV data from Yahoo Finance via
`yfinance`, with local CSV caching (so repeated runs don't re-hit the
network) and retry logic with exponential backoff for flaky downloads.
"""

from __future__ import annotations

import os
import time

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # yfinance is only required at call time, not import time
    yf = None


class DataLoader:
    """Fetches and caches OHLCV price data for one or more tickers."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str, interval: str) -> str:
        safe_ticker = ticker.replace("/", "_").replace("^", "idx_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{start}_{end}_{interval}.csv")

    def get_ohlcv(self, ticker: str, start: str, end: str, interval: str = "1d",
                   max_retries: int = 3, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker. Returns a DataFrame with
        columns [Open, High, Low, Close, Volume], indexed by date.

        Raises a clear error (rather than returning an empty/garbage
        frame) if the ticker is invalid or the network call ultimately
        fails after retries.
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")

        cache_path = self._cache_path(ticker, start, end, interval)
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        if yf is None:
            raise ImportError(
                "yfinance is not installed in this environment. "
                "Run `!pip install yfinance` in a Colab cell first."
            )

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    ticker, start=start, end=end, interval=interval,
                    auto_adjust=True, progress=False,
                )
                if df is None or df.empty:
                    raise ValueError(
                        f"No data returned for ticker '{ticker}' between {start} and {end}. "
                        "Check the symbol is correct and the date range contains trading days."
                    )
                # yfinance can return a MultiIndex column structure for some calls; flatten it.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                df.to_csv(cache_path)
                return df
            except Exception as exc:  # noqa: BLE001 -- retry on any transient failure
                last_error = exc
                if attempt < max_retries:
                    wait_seconds = 2 ** attempt
                    print(f"[DataLoader] Attempt {attempt}/{max_retries} failed for '{ticker}': {exc}. "
                          f"Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"Failed to download data for '{ticker}' after {max_retries} attempts."
        ) from last_error

    def get_multiple(self, tickers: list[str], start: str, end: str,
                      interval: str = "1d", **kwargs) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for several tickers, skipping (and reporting) any that fail."""
        data = {}
        failed = []
        for ticker in tickers:
            try:
                data[ticker] = self.get_ohlcv(ticker, start, end, interval, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- continue collecting the rest
                print(f"[DataLoader] Skipping '{ticker}': {exc}")
                failed.append(ticker)
        if failed:
            print(f"[DataLoader] Completed with {len(failed)} failed ticker(s): {failed}")
        return data
