"""Market data access over public Yahoo Finance endpoints (no auth, no cookies).

This is the only module that touches the network. It returns clean, lowercase
OHLCV ``pandas.DataFrame`` objects so the rest of the package never has to know
where the data came from.
"""

from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf

# Yahoo serves gold *futures* (GC=F), not XAUUSD spot. To report levels in spot
# numbers we pull a live spot quote from gold-api.com (free, no key) and use the
# basis (futures - spot) to convert. Symbols treated as gold for this purpose:
GOLD_SYMBOLS = {"GC=F", "MGC=F"}
_SPOT_URL = "https://api.gold-api.com/price/XAU"
_spot_cache: dict[str, tuple[float, float]] = {}  # key -> (timestamp, price)


def get_spot_xauusd(ttl: int = 60) -> float:
    """Return the live XAUUSD spot price (USD/oz) from gold-api.com.

    Cached for ``ttl`` seconds. Raises on network/parse failure so callers can
    fall back to reporting futures-based levels.
    """
    now = time.time()
    hit = _spot_cache.get("xau")
    if hit and now - hit[0] < ttl:
        return hit[1]
    resp = requests.get(_SPOT_URL, timeout=8)
    resp.raise_for_status()
    price = float(resp.json()["price"])
    _spot_cache["xau"] = (now, price)
    return price

# Intervals Yahoo accepts. We surface them so the MCP tool can validate input
# and give the model a clear error instead of an opaque upstream failure.
VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}

VALID_PERIODS = {
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to lowercase open/high/low/close/volume columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    # Single-ticker download can still return a column MultiIndex; collapse it.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].dropna(how="all")
    out.index.name = "date"
    return out


def get_ohlcv(
    symbol: str,
    interval: str = "1d",
    period: str = "6mo",
) -> pd.DataFrame:
    """Fetch OHLCV candles for ``symbol``.

    Raises ``ValueError`` for bad arguments or when no data comes back (e.g. an
    unknown ticker), so callers get an actionable message.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Valid: {', '.join(sorted(VALID_INTERVALS))}"
        )
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Valid: {', '.join(sorted(VALID_PERIODS))}"
        )

    raw = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df = _normalize(raw)
    if df.empty:
        raise ValueError(
            f"No data for symbol '{symbol}' (interval={interval}, period={period}). "
            "Check the ticker (e.g. 'AAPL', 'BTC-USD', 'EURUSD=X')."
        )
    return df


@lru_cache(maxsize=256)
def get_quote(symbol: str) -> dict:
    """Latest snapshot for a symbol: price, day range, volume, market cap, etc."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    out = {
        "symbol": symbol.upper(),
        "last_price": getattr(info, "last_price", None),
        "previous_close": getattr(info, "previous_close", None),
        "open": getattr(info, "open", None),
        "day_high": getattr(info, "day_high", None),
        "day_low": getattr(info, "day_low", None),
        "year_high": getattr(info, "year_high", None),
        "year_low": getattr(info, "year_low", None),
        "volume": getattr(info, "last_volume", None),
        "market_cap": getattr(info, "market_cap", None),
        "currency": getattr(info, "currency", None),
        "exchange": getattr(info, "exchange", None),
    }
    last, prev = out["last_price"], out["previous_close"]
    if last is not None and prev:
        out["change"] = round(last - prev, 6)
        out["change_pct"] = round(100 * (last - prev) / prev, 4)
    return out
