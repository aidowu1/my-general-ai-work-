# mcp_datetime_server.py
from starlette.applications import Starlette
from starlette.routing import Mount
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import uvicorn
from datetime import datetime
import zoneinfo

import configs

class DatetimeCalculatorMcpService:
    """
    Datetime calculator service.
    This service provides the current datetime in a specified timezone.
    """

    mcp = FastMCP("DatetimeMCP")

    @staticmethod
    @mcp.tool()
    def get_current_datetime(timezone: str = "UTC") -> str:
        """Return ISO datetime in requested tz (fallback to UTC)."""
        try:
            tz = zoneinfo.ZoneInfo(timezone)
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        return datetime.now(tz).isoformat()



if __name__ == "__main__":
    DatetimeCalculatorMcpService.mcp.run(
        transport=configs.TRANSPORT_TYPE,
        host=configs.MCP_HOST,
        port=configs.CURRENCY_CONVERTER_PORT,  # Assuming same port as currency converter for simplicity
        path=configs.MCP_PATH,
        log_level="debug",
    )
