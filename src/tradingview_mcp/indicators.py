"""Technical indicators computed with pure pandas/numpy.

Every function takes an OHLCV ``pandas.DataFrame`` (columns: ``open``, ``high``,
``low``, ``close``, ``volume``) and returns either a ``Series`` or a ``DataFrame``
aligned to the input index. Keeping these dependency-free (no TA-Lib) makes the
package easy to install via ``uvx`` and easy to unit test without a network call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Moving averages -------------------------------------------------------


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    return close.rolling(length).mean()


def ema(close: pd.Series, length: int = 20) -> pd.Series:
    return close.ewm(span=length, adjust=False).mean()


def wma(close: pd.Series, length: int = 20) -> pd.Series:
    weights = np.arange(1, length + 1)
    return close.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(close: pd.Series, length: int = 20) -> pd.Series:
    """Hull moving average."""
    half = wma(close, max(1, length // 2))
    full = wma(close, length)
    return wma(2 * half - full, max(1, int(np.sqrt(length))))


# --- Momentum / oscillators ------------------------------------------------


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3
) -> pd.DataFrame:
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    raw_k = 100 * (df["close"] - low_k) / (high_k - low_k)
    k_line = raw_k.rolling(smooth_k).mean()
    d_line = k_line.rolling(d).mean()
    return pd.DataFrame({"%K": k_line, "%D": d_line})


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].rolling(length).max()
    low = df["low"].rolling(length).min()
    return -100 * (high - df["close"]) / (high - low)


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(length).mean()
    md = (tp - ma).abs().rolling(length).mean()
    return (tp - ma) / (0.015 * md)


def roc(close: pd.Series, length: int = 12) -> pd.Series:
    return 100 * (close / close.shift(length) - 1)


def momentum(close: pd.Series, length: int = 10) -> pd.Series:
    return close.diff(length)


# --- Volatility ------------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, length)
    dev = close.rolling(length).std(ddof=0)
    return pd.DataFrame(
        {"upper": mid + std * dev, "middle": mid, "lower": mid - std * dev}
    )


def keltner_channels(
    df: pd.DataFrame, length: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    mid = ema(df["close"], length)
    rng = atr(df, length)
    return pd.DataFrame(
        {"upper": mid + mult * rng, "middle": mid, "lower": mid - mult * rng}
    )


def donchian_channels(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(length).max()
    lower = df["low"].rolling(length).min()
    return pd.DataFrame({"upper": upper, "middle": (upper + lower) / 2, "lower": lower})


# --- Trend strength --------------------------------------------------------


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return pd.DataFrame({"adx": adx_, "+di": plus_di, "-di": minus_di})


# --- Volume ----------------------------------------------------------------


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = tp * df["volume"]
    delta = tp.diff()
    pos_flow = raw_flow.where(delta > 0, 0.0).rolling(length).sum()
    neg_flow = raw_flow.where(delta < 0, 0.0).rolling(length).sum()
    ratio = pos_flow / neg_flow
    return 100 - (100 / (1 + ratio))


# --- Dispatch table --------------------------------------------------------

# Maps an indicator name -> (callable, kind) where kind is "close" if the
# function expects a close Series, or "df" if it expects the full OHLCV frame.
_REGISTRY = {
    "sma": (sma, "close"),
    "ema": (ema, "close"),
    "wma": (wma, "close"),
    "hma": (hma, "close"),
    "rsi": (rsi, "close"),
    "roc": (roc, "close"),
    "momentum": (momentum, "close"),
    "macd": (macd, "close"),
    "bollinger": (bollinger_bands, "close"),
    "stochastic": (stochastic, "df"),
    "williams_r": (williams_r, "df"),
    "cci": (cci, "df"),
    "atr": (atr, "df"),
    "keltner": (keltner_channels, "df"),
    "donchian": (donchian_channels, "df"),
    "adx": (adx, "df"),
    "obv": (obv, "df"),
    "vwap": (vwap, "df"),
    "mfi": (mfi, "df"),
}

AVAILABLE_INDICATORS = sorted(_REGISTRY)


def compute(
    df: pd.DataFrame, name: str, **params
) -> pd.Series | pd.DataFrame:
    """Compute a single indicator by name with optional keyword parameters."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown indicator '{name}'. Available: {', '.join(AVAILABLE_INDICATORS)}"
        )
    func, kind = _REGISTRY[key]
    arg = df["close"] if kind == "close" else df
    return func(arg, **params)
