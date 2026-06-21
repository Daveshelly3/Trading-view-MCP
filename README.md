# TradingView MCP Server

A community [Model Context Protocol](https://modelcontextprotocol.io) server that
gives Claude Code (or any MCP client) TradingView-style market analysis:
OHLCV candles, a suite of 19 technical indicators, a flexible technical
**screener**, and a simple **backtester**.

Data is pulled from **public Yahoo Finance endpoints** — no API key, no account,
no session cookies. Low blast radius: it only performs read-only fetches of
public market data.

> ⚠️ **Not affiliated with TradingView.** Programmatic market-data collection can
> conflict with a provider's Terms of Use. This server avoids touching any
> TradingView account directly, but use it at your own discretion. Nothing here
> is financial advice.

## Features

| Tool | What it does |
| --- | --- |
| `get_price` | Latest snapshot: price, day range, volume, market cap, % change. |
| `get_candles` | OHLCV candles for any interval/period. |
| `list_indicators` | Lists every indicator the server can compute. |
| `get_indicators` | Computes one or more indicators (with custom params). |
| `screen_symbols` | Screens a list of tickers against technical conditions. |
| `backtest` | Long-only SMA-crossover backtest vs. buy & hold. |
| `list_market_options` | Valid intervals, periods, and ticker formats. |

**Indicators:** SMA, EMA, WMA, HMA, RSI, ROC, momentum, MACD, Bollinger Bands,
Stochastic, Williams %R, CCI, ATR, Keltner Channels, Donchian Channels, ADX,
OBV, VWAP, MFI.

**Tickers** follow Yahoo conventions: stocks `AAPL`, crypto `BTC-USD`,
forex `EURUSD=X`, indices `^GSPC`.

## Install

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Register the server with Claude Code (runs straight from this repo)
claude mcp add tradingview -- uvx --from /path/to/Trading-view-MCP tradingview-mcp
```

To make it available in every project (user scope) rather than just the current one:

```bash
claude mcp add -s user tradingview -- uvx --from /path/to/Trading-view-MCP tradingview-mcp
```

Once published to PyPI you'd instead use:

```bash
uv tool install tradingview-mcp-server
claude mcp add tradingview -- uvx --from tradingview-mcp-server tradingview-mcp
```

### Verify

```bash
claude mcp list      # tradingview should show "connected"
```

Inside a session, `/mcp` lists the live tools. Then just ask in plain language,
e.g. *"screen AAPL, MSFT, NVDA for RSI under 35 on the daily"* — Claude calls
the tools itself.

## 📱 Mobile bias web app

A one-button mobile web app that runs the multi-timeframe bias (15m, 1h, 4h,
daily, weekly) and prints **Bullish / Bearish / Neutral** for each, plus an
overall verdict.

```bash
uv run --extra web uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Then open `http://<your-computer-ip>:8000` on your phone (same Wi-Fi), or
`http://localhost:8000` on the same machine. Type a ticker (defaults to `GC=F`,
gold ≈ XAUUSD), tap **Run Analysis**, and you get a colour-coded board.

The same logic is exposed to Claude as the `get_bias` MCP tool, so you can also
just ask *"what's the bias on gold across timeframes?"*.

### How the bias is scored

Each timeframe casts 7 votes (price vs EMA20/EMA50, EMA20 vs EMA50, MACD
histogram, RSI, ADX direction, Stochastic). Score ≥ +2 → **Bullish**,
≤ −2 → **Bearish**, otherwise **Neutral**. It's a transparent tally, not a
black box — and it is *not* financial advice.

## Run it directly

```bash
uv run tradingview-mcp            # starts the stdio server
```

## Tool examples

**Screen for oversold names below their 200-day average:**

```json
{
  "symbols": ["AAPL", "MSFT", "NVDA", "TSLA"],
  "conditions": [
    {"indicator": "rsi", "params": {"length": 14}, "op": "<", "value": 35},
    {"indicator": "close", "op": "<",
     "value": {"indicator": "sma", "params": {"length": 200}}}
  ],
  "match": "all"
}
```

**Detect a 20/50 EMA bullish crossover:**

```json
{
  "symbols": ["BTC-USD", "ETH-USD"],
  "conditions": [
    {"indicator": "ema", "params": {"length": 20}, "op": "crosses_above",
     "indicator2": "ema", "params2": {"length": 50}}
  ]
}
```

Multi-column indicators (`macd`, `bollinger`, `stochastic`, `adx`,
`keltner`, `donchian`) take a `column`, e.g.
`{"indicator": "macd", "column": "histogram", "op": ">", "value": 0}`.

## Development

```bash
uv sync --extra dev
uv run pytest          # indicator + screener tests, no network needed
```

## Architecture

```
src/tradingview_mcp/
  server.py       # FastMCP tool definitions (the MCP surface)
  data.py         # the only module that touches the network (yfinance)
  indicators.py   # pure pandas/numpy indicator engine (TA-Lib-free)
  screener.py     # screening + backtesting on top of the engine
```

## License

MIT
