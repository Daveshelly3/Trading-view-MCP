"""Tiny FastAPI app: one button -> multi-timeframe bias for a symbol.

Run with:  uv run --extra web uvicorn webapp.app:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 on your phone (same Wi-Fi) or laptop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tradingview_mcp.bias import full_bias

app = FastAPI(title="Bias Board", docs_url=None, redoc_url=None)

_STATIC = Path(__file__).parent / "static"


@app.get("/api/bias")
def api_bias(symbol: str = "GC=F") -> JSONResponse:
    """Return the multi-timeframe bias for ``symbol`` (default gold futures)."""
    try:
        data = full_bias(symbol)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=502)
    data["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return JSONResponse(data)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
