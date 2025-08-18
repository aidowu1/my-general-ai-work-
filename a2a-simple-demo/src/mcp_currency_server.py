# mcp_currency_server.py
from starlette.applications import Starlette
from starlette.routing import Mount
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import uvicorn

mcp = FastMCP("CurrencyMCP")

@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Simple example: returns converted amount and the implied rate.
    (Production: call a real FX API and secure it.)
    """
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.80, "JPY": 150.0}
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency not in rates or to_currency not in rates:
        raise ValueError("unsupported currency")
    rate = rates[to_currency] / rates[from_currency]
    converted = amount * rate
    return {"converted": round(converted, 6), "rate": round(rate, 8)}

# Create the ASGI app
# mcp_app = mcp.http_app(path='/mcp')

# mount the FastMCP ASGI app at /mcp
#app = Starlette(routes=[Mount("/mcp", app=mcp.http_app(path='/mcp'))])

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8001)
    # uvicorn.run(mcp, host="127.0.0.1", port=8001)
    # mcp.run()
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001,
        path="/mcp",
        log_level="debug",
    )