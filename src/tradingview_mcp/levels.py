"""Key price levels per timeframe (support / resistance the price may trade to).

Levels are derived from a single OHLCV frame so each timeframe yields its own
set: tight levels on 15m, progressively wider ones on 4h / daily / weekly. An
``offset`` (the futures-vs-spot basis) is subtracted from every level so gold
levels can be reported in XAUUSD spot numbers rather than GC=F futures numbers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingview_mcp import indicators as ta


def _dedupe(levels: list[dict[str, Any]], eps: float) -> list[dict[str, Any]]:
    """Drop levels that sit within ``eps`` of one already kept (nearest wins)."""
    kept: list[dict[str, Any]] = []
    for lvl in levels:
        if all(abs(lvl["price"] - k["price"]) > eps for k in kept):
            kept.append(lvl)
    return kept


def key_levels(
    df: pd.DataFrame,
    offset: float = 0.0,
    swing: int = 20,
    max_each: int = 5,
) -> dict[str, Any]:
    """Compute support/resistance levels for one timeframe's candles.

    Args:
        df: OHLCV frame for the timeframe.
        offset: Subtracted from every price level (futures->spot basis).
        swing: Lookback (bars) for the swing high/low.
        max_each: Cap on how many support and resistance levels to return.
    """
    high, low, close = df["high"], df["low"], df["close"]
    current = float(close.iloc[-1])
    atr = float(ta.atr(df, 14).dropna().iloc[-1])

    # Classic pivots from the last *completed* bar (avoid the forming candle).
    bar = -2 if len(df) >= 2 else -1
    H, L, C = float(high.iloc[bar]), float(low.iloc[bar]), float(close.iloc[bar])
    P = (H + L + C) / 3
    rng = H - L

    bb = ta.bollinger_bands(close, 20, 2.0).dropna()
    bb_u = float(bb["upper"].iloc[-1])
    bb_l = float(bb["lower"].iloc[-1])

    candidates: list[tuple[str, float]] = [
        ("Swing High", float(high.iloc[-swing:].max())),
        ("R2", P + rng),
        ("R1", 2 * P - L),
        ("+2 ATR", current + 2 * atr),
        ("+1 ATR", current + atr),
        ("BB Upper", bb_u),
        ("Pivot", P),
        ("BB Lower", bb_l),
        ("-1 ATR", current - atr),
        ("-2 ATR", current - 2 * atr),
        ("S1", 2 * P - H),
        ("S2", P - rng),
        ("Swing Low", float(low.iloc[-swing:].min())),
    ]

    cur = round(current - offset, 2)
    # Dedup epsilon scales with volatility so we don't collapse distinct levels.
    eps = max(0.1, atr * 0.1)

    resistance, support = [], []
    for name, raw in candidates:
        price = round(raw - offset, 2)
        if price > cur:
            resistance.append({"name": name, "price": price, "dist": round(price - cur, 2)})
        elif price < cur:
            support.append({"name": name, "price": price, "dist": round(cur - price, 2)})

    resistance = _dedupe(sorted(resistance, key=lambda x: x["price"]), eps)[:max_each]
    support = _dedupe(sorted(support, key=lambda x: -x["price"]), eps)[:max_each]

    return {
        "current": cur,
        "atr": round(atr, 2),
        "resistance": resistance,  # nearest above first
        "support": support,        # nearest below first
    }
