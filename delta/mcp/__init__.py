"""MCP Server for Delta LTSC compression.

Exposes token compression as tools for MCP-compatible clients like Cursor,
Claude Desktop, and other AI coding assistants.

Usage:
    # Run the MCP server
    delta-mcp

    # Or via Python
    python -m delta.mcp
"""

from .config import MCPConfig
from .metrics import OperationMetrics, SessionStats, MetricsStore
from .server import run_server, create_server

__all__ = [
    "MCPConfig",
    "OperationMetrics",
    "SessionStats",
    "MetricsStore",
    "run_server",
    "create_server",
]
