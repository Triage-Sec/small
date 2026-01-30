# MCP Server for Small LTSC

Small LTSC provides an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that exposes token compression as tools for AI coding assistants like Cursor, Claude Desktop, Windsurf, and other MCP-compatible clients.

## Installation

```bash
# Install with MCP support
pip install "small-ltsc[mcp]"

# Or install all optional dependencies
pip install "small-ltsc[all,mcp]"
```

## Quick Start

### Running the Server

```bash
# Preferred (works reliably with venvs):
python -m small.mcp

# If `small-mcp` is on PATH (pip console script):
small-mcp
```

The server communicates via stdio (JSON-RPC over stdin/stdout), which is the standard MCP transport.

Note: On macOS, GUI apps often **don't inherit your shell PATH**. If your client logs show
`spawn small-mcp ENOENT`, configure the server with an **absolute path** to your Python
interpreter (or to `small-mcp`).

## Client Configuration

### Cursor

Add to your Cursor settings (`~/.cursor/mcp.json` or workspace `.cursor/mcp.json`).

Recommended configuration (absolute `python` path):

```json
{
  "mcpServers": {
    "small-ltsc": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "small.mcp"]
    }
  }
}
```

If you installed to a location already on PATH, this also works:

```json
{
  "mcpServers": {
    "small-ltsc": {
      "command": "small-mcp"
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "small-ltsc": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "small.mcp"]
    }
  }
}
```

### Windsurf

Add to Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "small-ltsc": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "small.mcp"]
    }
  }
}
```

### Custom MCP Client

For programmatic use:

```python
from small.mcp import create_server, MCPConfig

# Create with custom config
config = MCPConfig(
    max_input_tokens=50000,
    verify_roundtrip=True,
    log_level="DEBUG",
)
server = create_server(config)
server.run()
```

## Available Tools

### compress_tokens

Compress a sequence of LLM tokens using lossless LTSC compression.

**Parameters:**
- `tokens` (required): Array of token IDs to compress
- `min_length`: Minimum pattern length (default: 2)
- `max_length`: Maximum pattern length (default: 16)
- `selection_mode`: Algorithm - "greedy", "optimal", or "beam" (default: "greedy")

**Returns:** Compressed tokens, compression ratio, patterns found, timing

### decompress_tokens

Decompress a previously compressed token sequence.

**Parameters:**
- `tokens` (required): Compressed token sequence

**Returns:** Decompressed tokens, timing

### analyze_compression

Analyze compressibility without actually compressing.

**Parameters:**
- `tokens` (required): Token sequence to analyze

**Returns:** Potential savings, detected patterns, recommendation

### compress_text

Compress text by tokenizing with tiktoken then applying LTSC.

**Parameters:**
- `text` (required): Text to compress
- `encoding`: Tiktoken encoding (default: "cl100k_base")

**Returns:** Compressed tokens, statistics, timing breakdown

### compress_context

Compress an LLM context window with optional preservation of recent tokens.

**Parameters:**
- `context` (required): Full context window text
- `encoding`: Tiktoken encoding (default: "cl100k_base")
- `preserve_recent`: Tokens to keep uncompressed (default: 0)

**Returns:** Compressed tokens, cost estimates, timing

Cost estimates are input-token savings for these models (pricing as of 2026-01-30):
- GPT-5.2 Thinking (`gpt-5.2-thinking`)
- Gemini 3.0 Pro (`gemini-3.0-pro`)
- Gemini 3.0 Flash (`gemini-3.0-flash`)
- Claude Opus 4.5 (`claude-opus-4.5`)

### get_session_metrics

Get accumulated metrics for the current session.

**Parameters:**
- `include_operations`: Include per-operation details (default: false)

**Returns:** Session statistics, throughput, cost estimates

### get_historical_metrics

Load metrics from all previous sessions.

**Parameters:**
- `limit`: Max operations to return (default: 100)

**Returns:** Historical statistics and recent operations

### run_benchmark

Run compression benchmarks on test data.

**Parameters:**
- `size`: Token count for tests (default: 1000)
- `runs`: Runs per test case (default: 3)

**Returns:** Benchmark results for different input types

### reset_session_metrics

Reset current session metrics.

**Returns:** Previous session summary

## Configuration

All settings can be configured via environment variables (prefixed with `SMALL_MCP_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SMALL_MCP_MAX_INPUT_TOKENS` | 100000 | Max tokens per request |
| `SMALL_MCP_MAX_TEXT_LENGTH` | 500000 | Max text length (chars) |
| `SMALL_MCP_METRICS_DIR` | `~/.small` | Metrics storage directory |
| `SMALL_MCP_METRICS_FILE` | `mcp_metrics.jsonl` | Metrics filename |
| `SMALL_MCP_LOG_LEVEL` | INFO | Logging level |
| `SMALL_MCP_ENABLE_BENCHMARKS` | true | Allow benchmark tool |
| `SMALL_MCP_DEFAULT_MIN_LENGTH` | 2 | Default min pattern length |
| `SMALL_MCP_DEFAULT_MAX_LENGTH` | 16 | Default max pattern length |
| `SMALL_MCP_VERIFY_ROUNDTRIP` | true | Verify decompression |
| `SMALL_MCP_DISCOVERY_MODE` | suffix-array | Discovery algorithm |
| `SMALL_MCP_SELECTION_MODE` | greedy | Selection algorithm |

To disable metrics persistence, set `SMALL_MCP_METRICS_DIR=none`.

## Example Usage

### In Cursor/Claude

Once configured, you can ask the AI assistant to use compression:

> "Compress this JSON schema and tell me the savings"

> "Analyze how compressible my system prompt would be"

> "Show me session metrics for compression operations"

### Programmatic Testing

```python
import json
import subprocess

# Start server
proc = subprocess.Popen(
    ["small-mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

# Initialize
proc.stdin.write(json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"clientInfo": {"name": "test"}}
}) + "\n")
proc.stdin.flush()
response = json.loads(proc.stdout.readline())

# Call compress_tokens
proc.stdin.write(json.dumps({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "compress_tokens",
        "arguments": {"tokens": [1, 2, 3] * 100}
    }
}) + "\n")
proc.stdin.flush()
result = json.loads(proc.stdout.readline())
print(result)
```

## Metrics & Monitoring

The MCP server tracks all operations in a JSONL file (default: `~/.small/mcp_metrics.jsonl`). Each line contains:

```json
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "operation": "compress",
  "input_tokens": 1000,
  "output_tokens": 650,
  "compression_ratio": 0.65,
  "savings_percent": 35.0,
  "patterns_found": 12,
  "time_ms": 15.5,
  "success": true
}
```

Use `get_historical_metrics` to query this data or process the file directly for custom analysis.

## Troubleshooting

### Server won't start

1. Ensure `small-ltsc[mcp]` is installed
2. Check Python version (requires 3.10+)
3. Try running with debug logging: `SMALL_MCP_LOG_LEVEL=DEBUG small-mcp`

### Tools not appearing in client

1. Restart the MCP client after configuration changes
2. Verify the command path is correct
3. Check client-specific MCP documentation

### Compression not working

1. Ensure input has repeated patterns (random data won't compress)
2. Check token count limits
3. Try `analyze_compression` first to see potential savings

### Logs

Server logs go to stderr. In most MCP clients, these are captured in debug logs. Set `SMALL_MCP_LOG_LEVEL=DEBUG` for verbose output.
