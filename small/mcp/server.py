"""MCP Server implementation for Small LTSC.

Implements the Model Context Protocol (MCP) over stdio for integration
with AI coding assistants like Cursor, Claude Desktop, and Windsurf.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .. import __version__
from ..pattern_cache import PatternCache
from .config import MCPConfig
from .metrics import MetricsStore
from .tools import TOOL_DEFINITIONS, create_tool_handlers

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server that exposes LTSC compression tools."""

    def __init__(self, config: MCPConfig | None = None) -> None:
        """Initialize the MCP server.

        Args:
            config: Server configuration. If None, loads from environment.
        """
        self.config = config or MCPConfig.from_env()
        self.metrics = MetricsStore(self.config.metrics_path)
        self.pattern_cache = self._init_pattern_cache()
        self.tool_handlers = create_tool_handlers(
            self.config, self.metrics, self.pattern_cache
        )
        self._setup_logging()

    def _init_pattern_cache(self) -> PatternCache | None:
        """Initialize pattern cache if enabled."""
        if not self.config.enable_pattern_cache:
            return None

        cache = PatternCache(
            max_patterns=self.config.pattern_cache_max_patterns,
            min_frequency=2,
            decay_half_life=100,
        )

        # Load existing cache from disk if available
        cache_path = self.config.pattern_cache_path
        if cache_path and cache_path.exists():
            if cache.load(cache_path):
                logger.info("Loaded pattern cache: %d patterns", len(cache))

        return cache

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )

    def send_response(
        self,
        request_id: int | str | None,
        result: Any = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC response."""
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error:
            response["error"] = error
        else:
            response["result"] = result
        self._write(response)

    def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        self._write(notification)

    def _write(self, message: dict[str, Any]) -> None:
        """Write a message to stdout."""
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    def handle_initialize(
        self, request_id: int | str | None, params: dict[str, Any]
    ) -> None:
        """Handle the initialize request."""
        logger.info(
            "MCP client connected: %s",
            params.get("clientInfo", {}).get("name", "unknown"),
        )

        self.send_response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "small-ltsc",
                    "version": __version__,
                },
            },
        )

    def handle_tools_list(self, request_id: int | str | None) -> None:
        """Handle tools/list request."""
        self.send_response(request_id, {"tools": TOOL_DEFINITIONS})

    def handle_tools_call(
        self, request_id: int | str | None, params: dict[str, Any]
    ) -> None:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = self.tool_handlers.get(tool_name)
        if not handler:
            logger.warning("Unknown tool requested: %s", tool_name)
            self.send_response(
                request_id,
                error={"code": -32601, "message": f"Unknown tool: {tool_name}"},
            )
            return

        try:
            logger.debug("Calling tool %s with args: %s", tool_name, tool_args)
            result = handler(tool_args)
            self.send_response(
                request_id,
                {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            )
        except (ValueError, TypeError) as e:
            logger.warning("Validation error in %s: %s", tool_name, e)
            self.send_response(
                request_id,
                error={"code": -32602, "message": str(e)},
            )
        except Exception as e:
            logger.exception("Error in tool %s", tool_name)
            self.send_response(
                request_id,
                error={"code": -32000, "message": f"Internal error: {e}"},
            )

    def handle_request(self, request: dict[str, Any]) -> None:
        """Route a JSON-RPC request to the appropriate handler."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        logger.debug("Received request: method=%s, id=%s", method, request_id)

        # Initialize
        if method == "initialize":
            self.handle_initialize(request_id, params)
            return

        # Initialized notification (no response)
        if method == "notifications/initialized":
            logger.info("MCP session initialized")
            return

        # List tools
        if method == "tools/list":
            self.handle_tools_list(request_id)
            return

        # Call tool
        if method == "tools/call":
            self.handle_tools_call(request_id, params)
            return

        # Ping (health check)
        if method == "ping":
            self.send_response(request_id, {})
            return

        # Resources (not implemented but required by protocol)
        if method == "resources/list":
            self.send_response(request_id, {"resources": []})
            return

        if method == "resources/read":
            self.send_response(
                request_id,
                error={"code": -32601, "message": "Resources not supported"},
            )
            return

        # Prompts (not implemented)
        if method == "prompts/list":
            self.send_response(request_id, {"prompts": []})
            return

        # Unknown method
        logger.warning("Unknown method: %s", method)
        self.send_response(
            request_id,
            error={"code": -32601, "message": f"Method not found: {method}"},
        )

    def run(self) -> None:
        """Run the MCP server, reading from stdin."""
        logger.info("Small LTSC MCP server starting (version %s)", __version__)
        logger.info("Metrics: %s", self.config.metrics_path or "disabled")
        logger.info(
            "Pattern cache: %s (%d patterns)",
            "enabled" if self.pattern_cache else "disabled",
            len(self.pattern_cache) if self.pattern_cache else 0,
        )
        logger.info(
            "Max tokens: %d, Max text: %d chars",
            self.config.max_input_tokens,
            self.config.max_text_length,
        )

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                    self.handle_request(request)
                except json.JSONDecodeError as e:
                    logger.error("JSON parse error: %s", e)
                    # Can't send proper response without request_id
                except Exception as e:
                    logger.exception("Unexpected error handling request: %s", e)
        finally:
            # Save pattern cache on shutdown
            self._save_pattern_cache()

    def _save_pattern_cache(self) -> None:
        """Save pattern cache to disk."""
        if self.pattern_cache is None:
            return

        cache_path = self.config.pattern_cache_path
        if cache_path:
            try:
                self.pattern_cache.save(cache_path)
                logger.info("Saved pattern cache: %d patterns", len(self.pattern_cache))
            except OSError as e:
                logger.warning("Failed to save pattern cache: %s", e)


def create_server(config: MCPConfig | None = None) -> MCPServer:
    """Create an MCP server instance.

    Args:
        config: Optional configuration. Loads from env if not provided.

    Returns:
        Configured MCPServer instance.
    """
    return MCPServer(config)


def run_server(config: MCPConfig | None = None) -> None:
    """Create and run the MCP server.

    Args:
        config: Optional configuration. Loads from env if not provided.
    """
    server = create_server(config)
    server.run()
