# agent_b.py
from fastapi import FastAPI
import uvicorn
from typing import Any, Dict
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp import types as mcp_types

import configs

class AgentB:
    """
    Agent B is a simple A2A server that can convert currency and return the current datetime.
    It uses MCP tools for these functionalities.
    """

    app = FastAPI(title="Agent_B (A2A server)")

    @staticmethod
    @app.get("/.well-known/agent.json")
    def agent_card():
        # A minimal "AgentCard" style discovery endpoint (A2A agents usually publish one)
        return {
            "id": "agent_b",
            "name": "Agent B - currency+time",
            "description": "Converts currency and returns the current datetime (demo)",
            "skills": ["convert_currency", "get_current_datetime"],
        }

    @staticmethod
    async def call_mcp_tool(server_url: str, tool_name: str, arguments: Dict[str, Any]):
        """
        Connect to a Streamable HTTP MCP server and call a tool.
        server_url should be e.g. "http://127.0.0.1:8001/mcp"
        """
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                # Prefer structuredContent if present:
                if hasattr(result, "structuredContent") and result.structuredContent:
                    return result.structuredContent
                # Fallback to text content
                for c in result.content:
                    if isinstance(c, mcp_types.TextContent):
                        return {"text": c.text}
                return {"raw": str(result)}

    @staticmethod
    @app.post("/a2a/message")
    async def handle_message(payload: Dict[str, Any]):
        """
        Expect a JSON payload:
        {
        "task": {
            "action": "convert_currency",
            "amount": 100,
            "from": "USD",
            "to": "EUR"
        }
        }
        Response is JSON with results from the MCP tools.
        """
        currency_task = payload.get(configs.CURRENCY_CONVERTER_TASK_NAME, {})
        datetime_task = payload.get(configs.DATETIME_CALCULATOR_TASK_NAME, {})
        try:
            amount = float(currency_task["amount"])
            frm = str(currency_task["from"])
            to = str(currency_task["to"])            
        except Exception as e:
            return {"status": "error", "message": f"bad input: {e}"}

        # Call currency MCP server    
        currency = await AgentB.call_mcp_tool(
            configs.CURRENCY_CONVERTER_ENDPOINT, 
            configs.CURRENCY_CONVERTER_TOOL_NAME,
            {"amount": amount, "from_currency": frm, "to_currency": to}
        )

        # Call datetime MCP server
        dt = await AgentB.call_mcp_tool(
            configs.DATETIME_CALCULATOR_ENDPOINT, 
            configs.DATETIME_CALCULATOR_TOOL_NAME,
            {
                "timezone": datetime_task.get("timezone", "UTC")  # Default to UTC if not provided
            }
        )

        return {"status": "ok", "result": {"currency": currency, "datetime": dt}}

    @staticmethod
    def start_agent_b():
        """
        Start the Agent B server.
        """
        uvicorn.run(AgentB.app, host=configs.AGENT_B_HOST, port=configs.AGENT_B_PORT)


if __name__ == "__main__":
    AgentB.start_agent_b()
