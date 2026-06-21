"""Vercel serverless function: GET /api/bias.

Vercel serves ``public/index.html`` from its CDN and routes ``/api/bias`` to this
ASGI app (see ``vercel.json``). The analysis package lives in ``src/`` and is
bundled via ``includeFiles``; we add it to ``sys.path`` so there is a single
source of truth shared with the MCP server and local dev app.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Make the bundled ``src/tradingview_mcp`` package importable in the function.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from tradingview_mcp.bias import full_bias  # noqa: E402

app = FastAPI(docs_url=None, redoc_url=None)


@app.get("/api/bias")
def api_bias(symbol: str = "GC=F") -> JSONResponse:
    try:
        data = full_bias(symbol)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=502)
    data["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return JSONResponse(data)
