# mcp_datetime_server.py
from starlette.applications import Starlette
from starlette.routing import Mount
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import uvicorn
from datetime import datetime
import zoneinfo

mcp = FastMCP("DatetimeMCP")

@mcp.tool()
def get_current_datetime(timezone: str = "UTC") -> str:
    """Return ISO datetime in requested tz (fallback to UTC)."""
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    return datetime.now(tz).isoformat()

# app = Starlette(routes=[Mount("/mcp", app=mcp.http_app(path='/mcp'))])

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8002)
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8002,
        path="/mcp",
        log_level="debug",
    )
