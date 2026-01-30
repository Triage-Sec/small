"""Tests for MCP server and tool handlers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from small.mcp.config import MCPConfig
from small.mcp.metrics import MetricsStore, OperationMetrics, SessionStats
from small.mcp.tools import ToolHandlers, TOOL_DEFINITIONS, create_tool_handlers


@pytest.fixture
def config() -> MCPConfig:
    """Create test configuration with no persistence."""
    return MCPConfig(
        max_input_tokens=10000,
        max_text_length=50000,
        metrics_dir=None,  # Disable persistence
        verify_roundtrip=True,
    )


@pytest.fixture
def metrics_store(config: MCPConfig) -> MetricsStore:
    """Create metrics store without persistence."""
    return MetricsStore(metrics_path=None)


@pytest.fixture
def handlers(config: MCPConfig, metrics_store: MetricsStore) -> ToolHandlers:
    """Create tool handlers for testing."""
    return ToolHandlers(config, metrics_store)


class TestMCPConfig:
    """Tests for MCPConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MCPConfig()
        assert config.max_input_tokens == 100_000
        assert config.max_text_length == 500_000
        assert config.log_level == "INFO"
        assert config.verify_roundtrip is True
        assert config.discovery_mode == "suffix-array"
        assert config.selection_mode == "greedy"

    def test_metrics_path(self) -> None:
        """Test metrics path construction."""
        config = MCPConfig(metrics_dir=Path("/tmp/test"), metrics_file="test.jsonl")
        assert config.metrics_path == Path("/tmp/test/test.jsonl")

    def test_metrics_path_none(self) -> None:
        """Test metrics path when disabled."""
        config = MCPConfig(metrics_dir=None)
        assert config.metrics_path is None


class TestMetricsStore:
    """Tests for MetricsStore."""

    def test_record_compress(self, metrics_store: MetricsStore) -> None:
        """Test recording compression metrics."""
        metrics = OperationMetrics(
            timestamp="2024-01-01T00:00:00",
            operation="compress",
            input_tokens=100,
            output_tokens=70,
            compression_ratio=0.7,
            savings_percent=30.0,
            patterns_found=5,
            time_ms=10.0,
            success=True,
        )
        metrics_store.record(metrics)

        session = metrics_store.session
        assert session.total_operations == 1
        assert session.compress_operations == 1
        assert session.total_input_tokens == 100
        assert session.total_output_tokens == 70
        assert session.total_tokens_saved == 30

    def test_record_decompress(self, metrics_store: MetricsStore) -> None:
        """Test recording decompression metrics."""
        metrics = OperationMetrics(
            timestamp="2024-01-01T00:00:00",
            operation="decompress",
            input_tokens=70,
            output_tokens=100,
            compression_ratio=0.7,
            savings_percent=0,
            patterns_found=0,
            time_ms=5.0,
            success=True,
        )
        metrics_store.record(metrics)

        session = metrics_store.session
        assert session.total_operations == 1
        assert session.decompress_operations == 1

    def test_persistence(self) -> None:
        """Test metrics persistence to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            store = MetricsStore(metrics_path)

            metrics = OperationMetrics(
                timestamp="2024-01-01T00:00:00",
                operation="compress",
                input_tokens=100,
                output_tokens=70,
                compression_ratio=0.7,
                savings_percent=30.0,
                patterns_found=5,
                time_ms=10.0,
                success=True,
            )
            store.record(metrics)

            # Verify file was written
            assert metrics_path.exists()
            with open(metrics_path) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["operation"] == "compress"
                assert data["input_tokens"] == 100

    def test_reset(self, metrics_store: MetricsStore) -> None:
        """Test session reset."""
        metrics = OperationMetrics(
            timestamp="2024-01-01T00:00:00",
            operation="compress",
            input_tokens=100,
            output_tokens=70,
            compression_ratio=0.7,
            savings_percent=30.0,
            patterns_found=5,
            time_ms=10.0,
            success=True,
        )
        metrics_store.record(metrics)

        old_session = metrics_store.reset()
        assert old_session["total_operations"] == 1
        assert metrics_store.session.total_operations == 0


class TestSessionStats:
    """Tests for SessionStats."""

    def test_avg_compression_ratio(self) -> None:
        """Test average compression ratio calculation."""
        stats = SessionStats()
        stats._ratios = [0.5, 0.6, 0.7]
        assert stats.avg_compression_ratio == pytest.approx(0.6)

    def test_avg_compression_ratio_empty(self) -> None:
        """Test average compression ratio with no data."""
        stats = SessionStats()
        assert stats.avg_compression_ratio == 1.0

    def test_best_worst_savings(self) -> None:
        """Test best and worst savings tracking."""
        stats = SessionStats()
        stats._savings = [10.0, 30.0, 20.0]
        assert stats.best_savings_percent == 30.0
        assert stats.worst_savings_percent == 10.0


class TestToolHandlers:
    """Tests for tool handlers."""

    def test_compress_tokens_basic(self, handlers: ToolHandlers) -> None:
        """Test basic token compression."""
        # Create repetitive input
        tokens = [1, 2, 3, 4, 5] * 20

        result = handlers.compress_tokens({"tokens": tokens})

        assert "compressed_tokens" in result
        assert result["original_length"] == 100
        assert result["compressed_length"] < result["original_length"]
        assert result["savings_percent"] > 0

    def test_compress_tokens_no_compression(self, handlers: ToolHandlers) -> None:
        """Test compression of random data (no patterns)."""
        tokens = list(range(100))  # No repetition

        result = handlers.compress_tokens({"tokens": tokens})

        # Should still succeed, just with no savings
        assert "compressed_tokens" in result
        assert result["original_length"] == 100

    def test_decompress_tokens(self, handlers: ToolHandlers) -> None:
        """Test token decompression."""
        # First compress
        original = [1, 2, 3, 4, 5] * 20
        compressed = handlers.compress_tokens({"tokens": original})

        # Then decompress
        result = handlers.decompress_tokens(
            {"tokens": compressed["compressed_tokens"]}
        )

        assert result["decompressed_tokens"] == original

    def test_analyze_compression(self, handlers: ToolHandlers) -> None:
        """Test compression analysis."""
        tokens = [1, 2, 3] * 50

        result = handlers.analyze_compression({"tokens": tokens})

        assert "potential_savings_percent" in result
        assert "patterns_detected" in result
        assert "recommendation" in result
        assert result["recommendation"] in [
            "highly_recommended",
            "recommended", 
            "marginal",
            "not_recommended",
        ]

    def test_compress_text(self, handlers: ToolHandlers) -> None:
        """Test text compression."""
        text = "Hello world! " * 50

        result = handlers.compress_text({"text": text})

        assert "compressed_tokens" in result
        assert "original_token_count" in result
        assert "timing" in result

    def test_compress_context(self, handlers: ToolHandlers) -> None:
        """Test context window compression."""
        context = "System: You are a helpful assistant.\n" * 20

        result = handlers.compress_context({"context": context})

        assert "compressed_tokens" in result
        assert "cost_estimate" in result
        assert "estimated_input_saved_usd" in result["cost_estimate"]

        saved = result["cost_estimate"]["estimated_input_saved_usd"]
        assert "gpt-5.2-thinking" in saved
        assert "gemini-3.0-pro" in saved
        assert "gemini-3.0-flash" in saved
        assert "claude-opus-4.5" in saved

    def test_compress_context_preserve_recent(self, handlers: ToolHandlers) -> None:
        """Test context compression with preserved recent tokens."""
        context = "Hello world! " * 100

        result = handlers.compress_context({
            "context": context,
            "preserve_recent": 50,
        })

        assert result["preserved_token_count"] == 50

    def test_get_session_metrics(
        self, handlers: ToolHandlers, metrics_store: MetricsStore
    ) -> None:
        """Test session metrics retrieval."""
        # Do some operations first
        handlers.compress_tokens({"tokens": [1, 2, 3] * 50})

        result = handlers.get_session_metrics({})

        assert result["total_operations"] == 1
        assert result["compress_operations"] == 1
        assert "avg_compression_ratio" in result

    def test_run_benchmark(self, handlers: ToolHandlers) -> None:
        """Test benchmark execution."""
        result = handlers.run_benchmark({"size": 500, "runs": 2})

        assert "results" in result
        assert len(result["results"]) == 3  # repeated, code, random
        assert "summary" in result

    def test_reset_session_metrics(
        self, handlers: ToolHandlers, metrics_store: MetricsStore
    ) -> None:
        """Test session reset via handler."""
        handlers.compress_tokens({"tokens": [1, 2, 3] * 50})

        result = handlers.reset_session_metrics({})

        assert result["reset"] is True
        assert result["previous_session"]["total_operations"] == 1
        assert metrics_store.session.total_operations == 0


class TestToolValidation:
    """Tests for input validation."""

    def test_empty_tokens_rejected(self, handlers: ToolHandlers) -> None:
        """Test that empty token arrays are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            handlers.compress_tokens({"tokens": []})

    def test_token_limit_enforced(self, config: MCPConfig) -> None:
        """Test that token limits are enforced."""
        # Create config with low limit
        strict_config = MCPConfig(max_input_tokens=100, metrics_dir=None)
        store = MetricsStore(None)
        handlers = ToolHandlers(strict_config, store)

        with pytest.raises(ValueError, match="exceeds maximum"):
            handlers.compress_tokens({"tokens": list(range(200))})

    def test_text_limit_enforced(self, config: MCPConfig) -> None:
        """Test that text length limits are enforced."""
        strict_config = MCPConfig(max_text_length=100, metrics_dir=None)
        store = MetricsStore(None)
        handlers = ToolHandlers(strict_config, store)

        with pytest.raises(ValueError, match="exceeds maximum"):
            handlers.compress_text({"text": "x" * 200})


class TestToolDefinitions:
    """Tests for tool definition schemas."""

    def test_all_tools_have_required_fields(self) -> None:
        """Test that all tool definitions have required MCP fields."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_handler_exists_for_each_tool(
        self, config: MCPConfig, metrics_store: MetricsStore
    ) -> None:
        """Test that each tool definition has a corresponding handler."""
        handlers = create_tool_handlers(config, metrics_store)
        tool_names = {t["name"] for t in TOOL_DEFINITIONS}
        handler_names = set(handlers.keys())
        assert tool_names == handler_names


class TestCreateToolHandlers:
    """Tests for create_tool_handlers factory."""

    def test_creates_all_handlers(
        self, config: MCPConfig, metrics_store: MetricsStore
    ) -> None:
        """Test that factory creates all expected handlers."""
        handlers = create_tool_handlers(config, metrics_store)

        expected = {
            "compress_tokens",
            "decompress_tokens",
            "analyze_compression",
            "compress_text",
            "compress_context",
            "get_session_metrics",
            "get_historical_metrics",
            "run_benchmark",
            "reset_session_metrics",
        }
        assert set(handlers.keys()) == expected

    def test_handlers_are_callable(
        self, config: MCPConfig, metrics_store: MetricsStore
    ) -> None:
        """Test that all handlers are callable."""
        handlers = create_tool_handlers(config, metrics_store)
        for name, handler in handlers.items():
            assert callable(handler), f"{name} is not callable"
