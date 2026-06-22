"""Multi-timeframe bias scoring.

Turns the raw indicators into a single verdict per timeframe: ``Bullish``,
``Bearish`` or ``Neutral``. The score is a transparent vote tally across trend,
momentum and directional-strength signals so the answer is explainable rather
than a black box.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from tradingview_mcp import indicators as ta
from tradingview_mcp.data import GOLD_SYMBOLS, get_ohlcv, get_spot_xauusd
from tradingview_mcp.levels import key_levels

# Time-to-live (seconds) for cached bias results. Intraday candles update far
# slower than this, so a short TTL keeps results fresh while shielding Yahoo from
# repeated taps. Override with the BIAS_CACHE_TTL env var.
CACHE_TTL = int(os.environ.get("BIAS_CACHE_TTL", "60"))

# label, interval, lookback period. Periods are chosen so EMA50/ADX(14) always
# have enough bars to be valid on every timeframe.
TIMEFRAMES: list[tuple[str, str, str]] = [
    ("Next 15 min", "15m", "1mo"),
    ("Next Hour", "1h", "3mo"),
    ("Next 4 Hours", "1h", "6mo"),  # 1h data, resampled to 4h below
    ("Daily", "1d", "1y"),
    ("Weekly", "1wk", "5y"),
]

# Resample rule applied when a timeframe needs a coarser candle than Yahoo serves.
_RESAMPLE = {"Next 4 Hours": "4h"}


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(cols).dropna(how="any")


def score_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Score one OHLCV frame into a bias verdict with its component signals."""
    c = df["close"]
    last = float(c.iloc[-1])

    ema20 = float(ta.ema(c, 20).iloc[-1])
    ema50 = float(ta.ema(c, 50).iloc[-1])
    rsi = float(ta.rsi(c).iloc[-1])
    hist = float(ta.macd(c)["histogram"].iloc[-1])
    adx_row = ta.adx(df).iloc[-1]
    adx = float(adx_row["adx"])
    plus_di = float(adx_row["+di"])
    minus_di = float(adx_row["-di"])
    st = ta.stochastic(df).iloc[-1]
    k, d = float(st["%K"]), float(st["%D"])

    signals: dict[str, int] = {
        "price_vs_ema20": 1 if last > ema20 else -1,
        "price_vs_ema50": 1 if last > ema50 else -1,
        "ema20_vs_ema50": 1 if ema20 > ema50 else -1,
        "macd_histogram": 1 if hist > 0 else -1,
        "rsi": 1 if rsi > 55 else (-1 if rsi < 45 else 0),
        # Directional movement only counts when a trend actually exists.
        "adx_direction": (1 if plus_di > minus_di else -1) if adx > 20 else 0,
        "stochastic": 1 if k > d else -1,
    }

    score = sum(signals.values())
    max_score = len(signals)
    if score >= 2:
        bias = "Bullish"
    elif score <= -2:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return {
        "bias": bias,
        "score": score,
        "max_score": max_score,
        "confidence": round(abs(score) / max_score, 2),
        "price": round(last, 2),
        "signals": signals,
        "metrics": {
            "rsi": round(rsi, 1),
            "macd_hist": round(hist, 2),
            "adx": round(adx, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "stoch_k": round(k, 1),
            "stoch_d": round(d, 1),
        },
    }


def bias_for_timeframe(
    symbol: str, label: str, interval: str, period: str, offset: float = 0.0
) -> dict[str, Any]:
    """Compute the bias + key levels for one labelled timeframe."""
    df = get_ohlcv(symbol, interval=interval, period=period)
    if label in _RESAMPLE:
        df = _resample(df, _RESAMPLE[label])
    result = score_frame(df)
    result["levels"] = key_levels(df, offset=offset)
    result.update({"timeframe": label, "interval": interval, "bars": len(df)})
    return result


def full_bias(symbol: str = "GC=F") -> dict[str, Any]:
    """Compute bias + key levels across every timeframe plus a consensus verdict.

    For gold (GC=F/MGC=F) levels are converted from futures to XAUUSD spot using
    a live spot quote; for other symbols levels stay in the instrument's price.
    """
    is_gold = symbol.upper() in GOLD_SYMBOLS
    spot: float | None = None
    if is_gold:
        try:
            spot = get_spot_xauusd()
        except Exception:  # noqa: BLE001 - fall back to futures-priced levels
            spot = None

    offset = 0.0
    offset_set = False
    timeframes = []
    score_sum = 0
    counted = 0
    for label, interval, period in TIMEFRAMES:
        try:
            df = get_ohlcv(symbol, interval=interval, period=period)
            if label in _RESAMPLE:
                df = _resample(df, _RESAMPLE[label])
            # Derive the futures->spot basis once, from the freshest frame.
            if spot is not None and not offset_set:
                offset = float(df["close"].iloc[-1]) - spot
                offset_set = True
            tf = score_frame(df)
            tf["levels"] = key_levels(df, offset=offset)
            tf.update({"timeframe": label, "interval": interval, "bars": len(df)})
            timeframes.append(tf)
            score_sum += tf["score"]
            counted += 1
        except Exception as exc:  # noqa: BLE001 - surface per-timeframe failures
            timeframes.append(
                {"timeframe": label, "interval": interval, "bias": "N/A", "error": str(exc)}
            )

    if counted:
        avg = score_sum / counted
        overall = "Bullish" if avg >= 1 else ("Bearish" if avg <= -1 else "Neutral")
    else:
        overall = "N/A"

    return {
        "symbol": symbol.upper(),
        "overall_bias": overall,
        "timeframes": timeframes,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "levels_currency": "XAUUSD spot" if (is_gold and spot is not None) else symbol.upper(),
        "spot": round(spot, 2) if spot is not None else None,
        "basis": round(offset, 2) if offset_set else None,
    }


class _TTLCache:
    """Tiny thread-safe time-to-live cache.

    On Vercel this only helps within a single warm function instance (the CDN
    layer does the heavy lifting via Cache-Control), but it also makes the local
    dev server and the MCP tool cheap to hammer.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get_or_compute(
        self, key: str, ttl: int, compute: Callable[[], Any]
    ) -> tuple[Any, bool]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and now - entry[0] < ttl:
                return entry[1], True
        # Compute outside the lock so a slow fetch doesn't block other symbols.
        value = compute()
        with self._lock:
            self._store[key] = (time.time(), value)
        return value, False


_CACHE = _TTLCache()


def cached_bias(symbol: str = "GC=F", ttl: int | None = None) -> dict[str, Any]:
    """Return :func:`full_bias` for a symbol, served from a TTL cache when warm.

    Adds a ``cached`` flag so callers can tell a fresh computation from a hit.
    """
    ttl = CACHE_TTL if ttl is None else ttl
    key = symbol.upper()
    data, hit = _CACHE.get_or_compute(key, ttl, lambda: full_bias(symbol))
    result = dict(data)
    result["cached"] = hit
    result["cache_ttl"] = ttl
    return result
