"""Local dev server: one button -> multi-timeframe bias for a symbol.

Serves the shared page from ``public/index.html`` (the same file Vercel hosts on
its CDN in production) and exposes the ``/api/bias`` endpoint.

Run with:  uv run --extra web uvicorn webapp.app:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 on your phone (same Wi-Fi) or laptop.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from tradingview_mcp.bias import CACHE_TTL, cached_bias

app = FastAPI(title="Bias Board", docs_url=None, redoc_url=None)

_PUBLIC = Path(__file__).resolve().parent.parent / "public"
_CACHE_HEADER = f"public, s-maxage={CACHE_TTL}, stale-while-revalidate={CACHE_TTL * 5}"


@app.get("/api/bias")
def api_bias(symbol: str = "GC=F") -> JSONResponse:
    """Return the multi-timeframe bias for ``symbol`` (default gold futures)."""
    try:
        data = cached_bias(symbol)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=502)
    return JSONResponse(data, headers={"Cache-Control": _CACHE_HEADER})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_PUBLIC / "index.html")
