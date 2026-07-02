"""A tiny stdio MCP server used by the smoke test and as a config example.

Run standalone:  python examples/echo_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the provided text."""
    return text


@mcp.tool()
def uppercase(text: str) -> str:
    """Return the text uppercased."""
    return text.upper()


if __name__ == "__main__":
    mcp.run()
