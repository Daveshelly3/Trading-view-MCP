"""Console entry point for the TradingView MCP server."""

from tradingview_mcp.server import mcp


def main() -> None:
    """Run the MCP server over stdio (the transport Claude Code uses)."""
    mcp.run()


if __name__ == "__main__":
    main()
