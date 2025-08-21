# mcp_currency_server.py
from starlette.applications import Starlette
from starlette.routing import Mount
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import uvicorn

import configs
import simple_logger

class CurrencyConverterMcpService:
    """
    Currency converter service.
    This is a simple example that uses hardcoded exchange rates.
    In production, you would call a real FX API and secure it.
    """

    logger = simple_logger.SimpleLogger().logger
    mcp = FastMCP("CurrencyMCP")

    @staticmethod
    @mcp.tool()
    def convert_currency(
        amount: float, 
        from_currency: str, 
        to_currency: str
    ) -> dict:
        """
        Simple example: returns converted amount and the implied rate.
        (Production: call a real FX API and secure it.)
        :param amount: Amount to convert
        :param from_currency: Currency to convert from (e.g. "USD")
        :param to_currency: Currency to convert to (e.g. "EUR")
        :return: Dictionary with converted amount and rate
        """
        #rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.80, "JPY": 150.0}
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        CurrencyConverterMcpService.logger.info(f"from_currency: {from_currency}, to_currency: {to_currency}, amount: {amount}")
        
        if from_currency not in configs.RATES or to_currency not in configs.RATES:
            raise ValueError("unsupported currency")
        rate = configs.RATES[to_currency] / configs.RATES[from_currency]
        print(f"Converting {amount} {from_currency} to {to_currency} at rate {rate}")
        converted = amount * rate
        return {"converted": round(converted, 6), "rate": round(rate, 8)}
    

if __name__ == "__main__":
    CurrencyConverterMcpService.mcp.run(
        transport=configs.TRANSPORT_TYPE,
        host=configs.MCP_HOST,
        port=configs.CURRENCY_CONVERTER_PORT,
        path=configs.MCP_PATH,
        log_level="debug",
    )