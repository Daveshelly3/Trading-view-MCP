"""FastMCP server exposing TradingView-style tools to Claude Code.

Data comes from public Yahoo Finance endpoints — no API key, no account, no
session cookies. Tickers follow Yahoo conventions: stocks ``AAPL``, crypto
``BTC-USD``, FX ``EURUSD=X``, indices ``^GSPC``.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from tradingview_mcp import indicators
from tradingview_mcp.bias import full_bias
from tradingview_mcp.data import VALID_INTERVALS, VALID_PERIODS, get_ohlcv, get_quote
from tradingview_mcp.screener import backtest_sma_cross, screen

mcp = FastMCP("tradingview")


def _round(value: Any) -> Any:
    try:
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


@mcp.tool()
def get_price(symbol: str) -> dict:
    """Get a latest-price snapshot for a symbol (price, day range, volume, market cap).

    Args:
        symbol: Yahoo ticker, e.g. 'AAPL', 'BTC-USD', 'EURUSD=X', '^GSPC'.
    """
    return get_quote(symbol)


@mcp.tool()
def get_candles(
    symbol: str,
    interval: str = "1d",
    period: str = "6mo",
    limit: int = 100,
) -> dict:
    """Get OHLCV candles for a symbol.

    Args:
        symbol: Yahoo ticker, e.g. 'AAPL', 'BTC-USD'.
        interval: One of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.
        period: Lookback window: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max.
        limit: Max number of most-recent candles to return (default 100).
    """
    df = get_ohlcv(symbol, interval=interval, period=period).tail(limit)
    candles = [
        {
            "date": idx.isoformat(),
            "open": _round(row["open"]),
            "high": _round(row["high"]),
            "low": _round(row["low"]),
            "close": _round(row["close"]),
            "volume": _round(row.get("volume")),
        }
        for idx, row in df.iterrows()
    ]
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "period": period,
        "count": len(candles),
        "candles": candles,
    }


@mcp.tool()
def list_indicators() -> dict:
    """List every technical indicator this server can compute."""
    return {"indicators": indicators.AVAILABLE_INDICATORS}


@mcp.tool()
def get_indicators(
    symbol: str,
    names: list[str],
    interval: str = "1d",
    period: str = "6mo",
    params: dict[str, dict] | None = None,
    last_n: int = 1,
) -> dict:
    """Compute one or more technical indicators for a symbol.

    Args:
        symbol: Yahoo ticker.
        names: Indicator names (see list_indicators), e.g. ['rsi','macd','ema'].
        interval: Candle interval (default '1d').
        period: Lookback window (default '6mo').
        params: Optional per-indicator params, e.g. {'rsi': {'length': 14}, 'ema': {'length': 50}}.
        last_n: How many trailing values to return per indicator (default 1 = latest only).
    """
    df = get_ohlcv(symbol, interval=interval, period=period)
    params = params or {}
    out: dict[str, Any] = {}
    for name in names:
        try:
            result = indicators.compute(df, name, **params.get(name, {}))
        except ValueError as exc:
            out[name] = {"error": str(exc)}
            continue
        tail = result.tail(last_n)
        if hasattr(tail, "columns"):  # DataFrame indicator (macd, bollinger, ...)
            out[name] = [
                {"date": idx.isoformat(), **{c: _round(r[c]) for c in tail.columns}}
                for idx, r in tail.iterrows()
            ]
        else:  # Series indicator
            out[name] = [
                {"date": idx.isoformat(), "value": _round(v)}
                for idx, v in tail.items()
            ]
    return {"symbol": symbol.upper(), "interval": interval, "indicators": out}


@mcp.tool()
def screen_symbols(
    symbols: list[str],
    conditions: list[dict],
    interval: str = "1d",
    period: str = "6mo",
    match: str = "all",
) -> dict:
    """Screen a list of symbols against technical conditions evaluated on the latest bar.

    Each condition is a dict. Examples:
        {"indicator": "rsi", "params": {"length": 14}, "op": "<", "value": 30}
        {"indicator": "close", "op": ">", "value": {"indicator": "sma", "params": {"length": 200}}}
        {"indicator": "ema", "params": {"length": 20}, "op": "crosses_above",
         "indicator2": "ema", "params2": {"length": 50}}

    Operators: >, >=, <, <=, ==, crosses_above, crosses_below.
    'value' may be a number or another indicator spec.
    For multi-column indicators (macd, bollinger, stochastic, adx) add "column",
    e.g. {"indicator": "macd", "column": "histogram", "op": ">", "value": 0}.

    Args:
        symbols: Tickers to screen, e.g. ['AAPL','MSFT','BTC-USD'].
        conditions: List of condition dicts (see above).
        interval: Candle interval (default '1d').
        period: Lookback window (default '6mo').
        match: 'all' (AND, default) or 'any' (OR).
    """
    results = screen(
        symbols, conditions, interval=interval, period=period, match=match
    )
    matched = [r["symbol"] for r in results if r.get("matched")]
    return {"matched": matched, "match_count": len(matched), "results": results}


@mcp.tool()
def backtest(
    symbol: str,
    fast: int = 20,
    slow: int = 50,
    interval: str = "1d",
    period: str = "2y",
    initial_cash: float = 10_000.0,
) -> dict:
    """Backtest a long-only SMA crossover strategy and compare to buy & hold.

    Returns total return, buy & hold return, number of trades, Sharpe ratio and
    max drawdown. Signals use the prior bar to avoid look-ahead bias.

    Args:
        symbol: Yahoo ticker.
        fast: Fast SMA length (default 20).
        slow: Slow SMA length (default 50).
        interval: Candle interval (default '1d').
        period: Lookback window (default '2y').
        initial_cash: Starting capital (default 10000).
    """
    return backtest_sma_cross(
        symbol,
        fast=fast,
        slow=slow,
        interval=interval,
        period=period,
        initial_cash=initial_cash,
    )


@mcp.tool()
def get_bias(symbol: str = "GC=F") -> dict:
    """Get a multi-timeframe technical bias (Bullish/Bearish/Neutral) for a symbol.

    Scores trend, momentum and directional-strength signals on each timeframe
    (15m, 1h, 4h, daily, weekly) plus an overall consensus verdict.

    Args:
        symbol: Yahoo ticker (default 'GC=F' = gold futures ~= XAUUSD).
    """
    return full_bias(symbol)


@mcp.tool()
def list_market_options() -> dict:
    """List valid intervals and periods accepted by the data tools."""
    return {
        "intervals": sorted(VALID_INTERVALS),
        "periods": sorted(VALID_PERIODS),
        "ticker_examples": {
            "stock": "AAPL",
            "crypto": "BTC-USD",
            "forex": "EURUSD=X",
            "index": "^GSPC",
        },
    }
