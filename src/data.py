"""
Data layer: download prices, align them, convert to returns.


"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for `tickers`.

    Returns a DataFrame indexed by date, one column per ticker.

    Caching: yfinance is slow and rate-limits. Once downloaded, prices are
    written to a parquet file and reused. Delete the cache file to refresh.
    """
    import yfinance as yf

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = f"{'_'.join(sorted(tickers))}_{start}_{end}".replace("-", "")
        cache_path = Path(cache_dir) / f"{key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,      # adjusts for splits AND dividends
        progress=False,
    )

    # yfinance returns a MultiIndex column frame for multiple tickers.
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        prices.columns = tickers

    prices = prices[sorted(tickers)]

    if cache_path is not None:
        prices.to_parquet(cache_path)

    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Align a multi-asset price panel onto a common set of trading days.

    Decisions made here, and why:
      - Drop any date where ANY asset is missing. Alternative is forward-fill,
        but forward-filling a price creates an artificial zero return, which
        understates volatility. For a VaR model that is the wrong direction of
        error, so we drop instead.
      - No interpolation, ever. Interpolated prices are invented data.
    """
    before = len(prices)
    cleaned = prices.dropna(how="any")
    dropped = before - len(cleaned)
    if dropped:
        print(f"[data] dropped {dropped} dates with missing prices "
              f"({dropped / before:.2%} of sample)")
    return cleaned


def to_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """
    Convert a price panel to returns.

    kind="log":    r_t = ln(P_t / P_{t-1})
    kind="simple": r_t = P_t / P_{t-1} - 1

    Log returns are time-additive (a 10-day log return is the sum of 10 daily
    log returns), which is what makes square-root-of-time scaling coherent.
    Simple returns are portfolio-additive (a portfolio's simple return is the
    weighted sum of asset simple returns). You cannot have both. State which
    one you chose in the README and be consistent.
    """
    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    elif kind == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"unknown return type: {kind!r}")
    return rets.dropna(how="any")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Collapse an asset return panel into a single portfolio return series
    using fixed weights (i.e. assume daily rebalancing back to target).

    Raises if weights don't match the columns or don't sum to 1.
    """
    missing = set(weights) - set(returns.columns)
    if missing:
        raise ValueError(f"weights reference tickers not in returns: {missing}")

    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"weights sum to {total}, expected 1.0")

    w = pd.Series(weights)[returns.columns]
    return returns @ w


def weight_vector(returns: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weights as a numpy array ordered to match the columns of `returns`."""
    return pd.Series(weights)[returns.columns].to_numpy()


def load_all(config_path: str = "config.yaml"):
    """
    One-call convenience loader.

    Returns (config, prices, asset_returns, portfolio_return_series).
    """
    cfg = load_config(config_path)
    tickers = list(cfg["portfolio"]["weights"].keys())

    prices = download_prices(
        tickers,
        cfg["data"]["start"],
        cfg["data"]["end"],
        cfg["data"].get("cache_dir"),
    )
    prices = clean_prices(prices)
    asset_rets = to_returns(prices, cfg["model"]["return_type"])
    port_rets = portfolio_returns(asset_rets, cfg["portfolio"]["weights"])

    return cfg, prices, asset_rets, port_rets


if __name__ == "__main__":
    cfg, prices, asset_rets, port_rets = load_all()
    print(f"\nprices:  {prices.shape[0]} days x {prices.shape[1]} assets")
    print(f"range:   {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\nportfolio daily return stats:")
    print(f"  mean:     {port_rets.mean():.6f}")
    print(f"  std:      {port_rets.std():.6f}")
    print(f"  ann. vol: {port_rets.std() * np.sqrt(252):.4f}")
    print(f"  skew:     {port_rets.skew():.4f}")
    print(f"  kurtosis: {port_rets.kurtosis():.4f}  (excess; normal = 0)")
    print(f"  worst:    {port_rets.min():.4f} on {port_rets.idxmin().date()}")
