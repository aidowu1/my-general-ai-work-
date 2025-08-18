# agent_b.py
from fastapi import FastAPI
import uvicorn
from typing import Any, Dict
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp import types as mcp_types

app = FastAPI(title="Agent_B (A2A server)")

@app.get("/.well-known/agent.json")
def agent_card():
    # A minimal "AgentCard" style discovery endpoint (A2A agents usually publish one)
    return {
        "id": "agent_b",
        "name": "Agent B - currency+time",
        "description": "Converts currency and returns the current datetime (demo)",
        "skills": ["convert_currency", "get_current_datetime"],
    }

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
    task = payload.get("task", {})
    if task.get("action") != "convert_currency":
        return {"status": "error", "message": "unsupported task"}

    try:
        amount = float(task["amount"])
        frm = str(task["from"])
        to = str(task["to"])
    except Exception as e:
        return {"status": "error", "message": f"bad input: {e}"}

    # Call currency MCP server
    currency = await call_mcp_tool("http://127.0.0.1:8001/mcp", "convert_currency",
                                  {"amount": amount, "from_currency": frm, "to_currency": to})

    # Call datetime MCP server
    dt = await call_mcp_tool("http://127.0.0.1:8002/mcp", "get_current_datetime",
                             {"timezone": "UTC"})

    return {"status": "ok", "result": {"currency": currency, "datetime": dt}}

# Run the FastAPI app
# This will start the server on http://
def start_agent_b():
    uvicorn.run(app, host="127.0.0.1", port=9001)

if __name__ == "__main__":
    start_agent_b()
