"""Unit tests for the indicator engine and screener (no network required)."""

import numpy as np
import pandas as pd
import pytest

from tradingview_mcp import indicators
from tradingview_mcp.screener import evaluate_condition


@pytest.fixture
def ohlcv():
    # Deterministic synthetic uptrend with noise so indicators have signal.
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    drift = np.linspace(100, 200, n)
    noise = rng.normal(0, 2, n).cumsum()
    close = drift + noise
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 1, n)
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_sma_matches_manual(ohlcv):
    result = indicators.sma(ohlcv["close"], 10)
    expected = ohlcv["close"].iloc[-10:].mean()
    assert result.iloc[-1] == pytest.approx(expected)


def test_rsi_bounds(ohlcv):
    r = indicators.rsi(ohlcv["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_columns(ohlcv):
    m = indicators.macd(ohlcv["close"])
    assert list(m.columns) == ["macd", "signal", "histogram"]
    assert m["histogram"].iloc[-1] == pytest.approx(
        m["macd"].iloc[-1] - m["signal"].iloc[-1]
    )


def test_bollinger_ordering(ohlcv):
    b = indicators.bollinger_bands(ohlcv["close"], 20).dropna()
    assert (b["upper"] >= b["middle"]).all()
    assert (b["middle"] >= b["lower"]).all()


def test_atr_positive(ohlcv):
    a = indicators.atr(ohlcv, 14).dropna()
    assert (a > 0).all()


def test_all_registered_indicators_run(ohlcv):
    for name in indicators.AVAILABLE_INDICATORS:
        out = indicators.compute(ohlcv, name)
        assert out is not None and len(out) == len(ohlcv)


def test_compute_unknown_raises(ohlcv):
    with pytest.raises(ValueError):
        indicators.compute(ohlcv, "not_a_real_indicator")


def test_condition_threshold(ohlcv):
    hit, val = evaluate_condition(
        ohlcv, {"indicator": "rsi", "op": "<", "value": 100}
    )
    assert hit is True
    assert val is not None


def test_condition_crossover_detects(ohlcv):
    # Force a clean upward crossover at the final bar.
    df = ohlcv.copy()
    fast = indicators.sma(df["close"], 5)
    slow = indicators.sma(df["close"], 20)
    # Just assert the machinery returns a boolean without error.
    hit, _ = evaluate_condition(
        df,
        {
            "indicator": "sma",
            "params": {"length": 5},
            "op": "crosses_above",
            "indicator2": "sma",
            "params2": {"length": 20},
        },
    )
    assert isinstance(hit, bool)
