"""Lightweight screening and backtesting built on the indicator engine."""

from __future__ import annotations

import operator
from typing import Any

import pandas as pd

from tradingview_mcp import indicators
from tradingview_mcp.data import get_ohlcv

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "crosses_above": None,  # handled specially below
    "crosses_below": None,
}


def _last_scalar(value: pd.Series | pd.DataFrame, column: str | None) -> float | None:
    if isinstance(value, pd.DataFrame):
        col = column or value.columns[0]
        series = value[col]
    else:
        series = value
    series = series.dropna()
    return None if series.empty else float(series.iloc[-1])


def _series(value: pd.Series | pd.DataFrame, column: str | None) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        return value[column or value.columns[0]]
    return value


def evaluate_condition(df: pd.DataFrame, cond: dict[str, Any]) -> tuple[bool, Any]:
    """Evaluate one screening condition against a symbol's OHLCV frame.

    A condition is a dict like::

        {"indicator": "rsi", "params": {"length": 14}, "op": "<", "value": 30}
        {"indicator": "close", "op": "crosses_above", "indicator2": "sma",
         "params2": {"length": 50}}

    ``close``/``open``/``high``/``low``/``volume`` are usable as pseudo-indicators.
    """
    op = cond["op"]
    if op not in _OPS:
        raise ValueError(f"Unknown op '{op}'. Valid: {', '.join(_OPS)}")

    left = _resolve(df, cond.get("indicator", "close"), cond.get("params"), cond.get("column"))

    if op in ("crosses_above", "crosses_below"):
        right = _resolve(
            df, cond["indicator2"], cond.get("params2"), cond.get("column2")
        )
        ls, rs = _series(left, cond.get("column")), _series(right, cond.get("column2"))
        joined = pd.concat([ls, rs], axis=1).dropna()
        if len(joined) < 2:
            return False, None
        a_prev, b_prev = joined.iloc[-2]
        a_now, b_now = joined.iloc[-1]
        if op == "crosses_above":
            hit = a_prev <= b_prev and a_now > b_now
        else:
            hit = a_prev >= b_prev and a_now < b_now
        return bool(hit), {"left": float(a_now), "right": float(b_now)}

    left_val = _last_scalar(left, cond.get("column"))
    if left_val is None:
        return False, None
    # value may be a literal number or another indicator spec
    if isinstance(cond.get("value"), dict):
        right = _resolve(
            df,
            cond["value"]["indicator"],
            cond["value"].get("params"),
            cond["value"].get("column"),
        )
        right_val = _last_scalar(right, cond["value"].get("column"))
    else:
        right_val = cond["value"]
    if right_val is None:
        return False, None
    return bool(_OPS[op](left_val, right_val)), left_val


def _resolve(df: pd.DataFrame, name: str, params: dict | None, column: str | None):
    name = name.lower()
    if name in ("open", "high", "low", "close", "volume"):
        return df[name]
    return indicators.compute(df, name, **(params or {}))


def screen(
    symbols: list[str],
    conditions: list[dict[str, Any]],
    interval: str = "1d",
    period: str = "6mo",
    match: str = "all",
) -> list[dict[str, Any]]:
    """Return symbols whose latest bar satisfies the conditions.

    ``match='all'`` requires every condition (AND); ``match='any'`` requires one (OR).
    """
    results = []
    for symbol in symbols:
        try:
            df = get_ohlcv(symbol, interval=interval, period=period)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
            results.append({"symbol": symbol, "matched": False, "error": str(exc)})
            continue
        outcomes, values = [], {}
        for cond in conditions:
            hit, val = evaluate_condition(df, cond)
            outcomes.append(hit)
            values[cond.get("indicator", "close")] = val
        matched = all(outcomes) if match == "all" else any(outcomes)
        results.append(
            {"symbol": symbol, "matched": matched, "values": values}
        )
    return results


def backtest_sma_cross(
    symbol: str,
    fast: int = 20,
    slow: int = 50,
    interval: str = "1d",
    period: str = "2y",
    initial_cash: float = 10_000.0,
) -> dict[str, Any]:
    """A simple long-only SMA crossover backtest (go long fast>slow, flat otherwise)."""
    df = get_ohlcv(symbol, interval=interval, period=period)
    fast_ma = indicators.sma(df["close"], fast)
    slow_ma = indicators.sma(df["close"], slow)
    # Position decided on prior bar's signal to avoid look-ahead bias.
    position = (fast_ma > slow_ma).astype(int).shift(1).fillna(0)
    returns = df["close"].pct_change().fillna(0)
    strat_returns = position * returns

    equity = (1 + strat_returns).cumprod() * initial_cash
    buy_hold = (1 + returns).cumprod() * initial_cash
    trades = int(position.diff().abs().sum())

    total_return = float(equity.iloc[-1] / initial_cash - 1)
    bh_return = float(buy_hold.iloc[-1] / initial_cash - 1)
    # Annualized Sharpe assuming the interval's bars; rough but useful.
    periods_per_year = {"1d": 252, "1wk": 52, "1mo": 12, "1h": 252 * 6.5}.get(
        interval, 252
    )
    std = strat_returns.std()
    sharpe = (
        float(strat_returns.mean() / std * (periods_per_year ** 0.5))
        if std > 0
        else 0.0
    )
    running_max = equity.cummax()
    max_drawdown = float(((equity - running_max) / running_max).min())

    return {
        "symbol": symbol.upper(),
        "strategy": f"SMA {fast}/{slow} crossover (long-only)",
        "interval": interval,
        "period": period,
        "initial_cash": initial_cash,
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(bh_return * 100, 2),
        "num_trades": trades,
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
    }
