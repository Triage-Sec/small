# Delta

**Lossless Token Sequence Compression for Large Language Models**

[![PyPI](https://img.shields.io/pypi/v/delta-ltsc)](https://pypi.org/project/delta-ltsc/)
[![npm](https://img.shields.io/npm/v/@delta-ltsc/sdk)](https://www.npmjs.com/package/@delta-ltsc/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

Delta is a production-ready lossless compression system that reduces the computational and economic cost of LLM inference by eliminating redundancy in input sequences before they reach the model. It replaces repeated multi-token patterns with compact meta-token references backed by a learnable dictionary format, achieving 30-60% compression on structured inputs while guaranteeing perfect reconstruction.

## About

### [Triage](https://triage-sec.com)

Triage is building the resiliency layer for AI systems. We expose deep observability into staging environments so coding agents can statically reason about codebases, predict runtime behavior, and remediate production issues autonomously. On the research side, we're filling latent space in frozen models to address known system weaknesses, building dynamic guardrails for real-time threat detection, and designing bureaucratic authorization flows that route sensitive operations to the right humans at the right time. The thesis behind everything we build: security should learn from every failure and compound over time, not reset with each new signature. We're compressing MTTR toward zero by treating every incident as training data for the next.

### Why We Built Delta

Delta emerged from a pattern we kept seeing while building Triage and talking to teams deploying LLM-powered products: context-augmented generation produces massively redundant token sequences, and inference costs scale linearly with every one of them. The same boilerplate, the same structural patterns, the same retrieved chunks appear repeatedly across requests. LTSC (Lossless Token Sequence Compression) identifies and compresses these recurring patterns without semantic loss, targeting the specific redundancy profiles that coding agents, retrieval systems, and multi-turn conversations create. We built it as an open-source contribution because inference cost remains one of the most underappreciated bottlenecks to AI adoption, particularly for the agentic workflows where context windows balloon quickly. Solving it at the compression layer lets teams ship more capable agents without burning through API budgets or making architectural compromises to stay under token limits.

### Contributors

This project was constructed by:
- **[Nikhil Srivastava](https://www.linkedin.com/in/srivastavan/)** (University of California, Berkeley)
- **[Omansh Bainsla](https://www.linkedin.com/in/omanshb/)** (Georgia Tech)
- **[Sahil Chatiwala](https://www.linkedin.com/in/sahil-chatiwala/)** (Georgia Tech)

## Why Delta?

As context augmentation techniques become standard practice (retrieval-augmented generation, tool schemas, code repositories, policy documents, multi-turn conversations), input sequences increasingly contain repeated subsequences that consume context window budget and quadratic attention compute without contributing new information.

**Delta addresses this by:**
- Compressing redundant patterns at the token level before inference
- Maintaining a learnable dictionary format that models can understand with minimal fine-tuning
- Guaranteeing lossless round-trip reconstruction

## Key Features

- **Lossless Compression**: Perfect round-trip reconstruction guaranteed via mathematical constraints
- **High Performance**: Rust/WASM core with O(n log n) suffix array algorithms
- **Cross-Platform**: Python library + TypeScript SDK for browsers, Node.js, Deno, and edge runtimes
- **Production Ready**: Comprehensive test suites, type safety, structured logging
- **Multiple Discovery Strategies**: Suffix array, BPE-style iterative, AST-aware (Python)
- **Optimal Selection**: Greedy, weighted interval scheduling, beam search, or ILP solvers
- **ML Integration**: Importance scoring, region-aware compression, quality prediction
- **Streaming Support**: Handle arbitrarily large inputs with constant memory

## Installation

### Python

```bash
pip install delta-ltsc

# With optional dependencies
pip install "delta-ltsc[analysis]"   # ML analysis tools
pip install "delta-ltsc[training]"   # Fine-tuning utilities
pip install "delta-ltsc[mcp]"        # MCP server for AI assistants
pip install "delta-ltsc[all]"        # Everything
```

### TypeScript/JavaScript

```bash
npm install @delta-ltsc/sdk

# Optional ML features
npm install @delta-ltsc/ml
```

## Quick Start

### Python

```python
from delta import compress, decompress, CompressionConfig

# Compress a token sequence
tokens = [101, 2054, 2003, 1996, 4248, 102] * 20  # Repeated pattern
config = CompressionConfig(verify=True)
result = compress(tokens, config)

print(f"Original: {result.original_length} tokens")
print(f"Compressed: {result.compressed_length} tokens")
print(f"Ratio: {result.compressed_length / result.original_length:.1%}")

# Decompress (lossless)
restored = decompress(result.serialized_tokens, config)
assert restored == tokens
```

### TypeScript

```typescript
import { compress, decompress, initWasm } from '@delta-ltsc/sdk';

// Initialize WASM (required once)
await initWasm();

// Compress tokens
const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
const result = await compress(tokens);

console.log(`Compressed: ${result.originalLength} → ${result.compressedLength} tokens`);
console.log(`Savings: ${((1 - result.compressionRatio) * 100).toFixed(1)}%`);

// Decompress (lossless)
const restored = await decompress(result.serializedTokens);
```

## How It Works

Delta identifies repeated token subsequences and replaces them with meta-tokens, storing the mapping in a prefix dictionary:

```
Original:  [the, cat, sat, on, the, mat, the, cat, ran]
                ^^^^^^^^            ^^^^^^^^
Compressed: [<Dict>, <MT_0>, <Len:2>, the, cat, </Dict>, <MT_0>, sat, on, the, mat, <MT_0>, ran]
```

The compression format is designed to be learnable by transformer models with minimal fine-tuning.

### Compressibility Constraint

A pattern is only compressed if it provides net savings:

```
length × count > 1 + length + count + overhead
```

This mathematical constraint ensures compression never increases sequence length.

## Configuration

### Python

```python
from delta import CompressionConfig

config = CompressionConfig(
    # Pattern discovery
    min_subsequence_length=2,
    max_subsequence_length=8,
    discovery_mode="suffix-array",  # "sliding-window", "bpe"
    
    # Selection algorithm
    selection_mode="greedy",        # "optimal", "beam", "ilp"
    beam_width=8,
    
    # Hierarchical compression
    hierarchical_enabled=True,
    hierarchical_max_depth=3,
    
    # ML features (optional)
    use_importance_scoring=False,
    enable_adaptive_regions=False,
    
    # Verification
    verify=True,
)
```

### TypeScript

```typescript
const result = await compress(tokens, {
  minSubsequenceLength: 2,
  maxSubsequenceLength: 8,
  selectionMode: 'greedy',      // 'optimal' | 'beam'
  hierarchicalEnabled: true,
  hierarchicalMaxDepth: 3,
  verify: true,
});
```

## Selection Algorithms

| Mode | Complexity | Description |
|------|------------|-------------|
| `greedy` | O(n log n) | Fast, savings-density heuristic |
| `optimal` | O(n²) | Weighted interval scheduling via DP |
| `beam` | O(n × width) | Beam search with marginal savings |
| `ilp` | Exponential | Globally optimal (requires scipy) |

## Advanced Features

### Static Dictionaries (TypeScript)

Use pre-built dictionaries for domain-specific content:

```typescript
const result = await compress(pythonCodeTokens, {
  staticDictionary: 'python-v1',
});
```

Available: `python-v1`, `typescript-v1`, `markdown-v1`, `json-v1`, `sql-v1`

### Streaming Compression

```typescript
import { createStreamingCompressor } from '@delta-ltsc/sdk';

const compressor = await createStreamingCompressor();
for await (const chunk of tokenStream) {
  await compressor.addChunk(chunk);
}
const result = await compressor.finish();
```

### Region-Aware Compression

```python
from delta import detect_regions, filter_candidates_by_region

# Detect semantic regions (SYSTEM, USER, CONTEXT, CODE)
regions = detect_regions(tokens)

# Apply per-region compression limits
filtered = filter_candidates_by_region(candidates, regions, tokens)
```

### Quality Prediction

```python
from delta import create_predictor

predictor = create_predictor(task_type="code")
prediction = predictor.predict(tokens, result)

if prediction.recommendation == "compress":
    # Safe to use compressed output
    pass
```

### MCP Integration (AI Assistants)

Delta provides an MCP server for integration with AI coding assistants:

```bash
# Install with MCP support
pip install "delta-ltsc[mcp]"

# Run the server
delta-mcp
```

Configure in Cursor/Claude Desktop (`~/.cursor/mcp.json`). Prefer **absolute paths** (GUI apps often don't inherit your shell `PATH`):

```json
{
  "mcpServers": {
    "delta-ltsc": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "delta.mcp"]
    }
  }
}
```

If you see `spawn delta-mcp ENOENT`, use an absolute path to your environment's `python` (or `delta-mcp`).

See [MCP Documentation](docs/mcp.md) for a full setup guide and available tools.

## Architecture

```
delta/                          # Python package
├── compressor.py               # Core compress/decompress API
├── config.py                   # Configuration dataclass
├── engine.py                   # Compression pipeline
├── discovery.py                # Pattern discovery algorithms
├── selection.py                # Pattern selection algorithms
├── adaptive.py                 # Region-aware compression
└── quality_predictor.py        # ML quality prediction

packages/
├── core/                       # Rust/WASM compression core
│   └── src/
│       ├── lib.rs              # WASM exports
│       ├── suffix_array.rs     # O(n log n) suffix array
│       ├── selection.rs        # Pattern selection
│       └── dictionary.rs       # Dictionary serialization
├── sdk/                        # TypeScript SDK
│   └── src/
│       ├── compress.ts         # High-level API
│       ├── streaming.ts        # Streaming support
│       └── worker.ts           # Worker thread support
└── ml/                         # Optional ML features
    └── src/
        ├── importance.ts       # Pattern importance scoring
        ├── quality.ts          # Quality prediction
        └── regions.ts          # Region detection
```

## Benchmarks

```bash
# Python benchmarks
python benchmarks/ratio.py --tokens 8192 --runs 10
python benchmarks/latency.py --tokens 8192 --runs 10

# TypeScript benchmarks
cd packages/sdk && npm run benchmark
```

Typical results on structured inputs:
- **Compression ratio**: 35-60% reduction
- **Latency**: < 10ms for 8K tokens (WASM), < 50ms (Python)
- **Memory**: O(n) with streaming support

## Testing

### Python

```bash
pytest                              # Run all tests
pytest --cov=delta --cov-report=html  # With coverage
```

### TypeScript

```bash
cd packages/sdk
npm test                            # Unit tests
npm run test:browser                # Browser tests
```

### Rust

```bash
cd packages/core
cargo test                          # Unit tests
```

## Documentation

- [Design Intent](docs/00-intent.md) - Motivation and objectives
- [Compression Format](docs/02-format.md) - Dictionary and body format specification
- [Architecture](docs/06-architecture.md) - System design overview
- [Algorithm Details](docs/ALGORITHMS.md) - Discovery and selection algorithms
- [ML Integration](docs/ML_INTEGRATION.md) - Importance scoring and quality prediction
- [API Reference](docs/API.md) - Complete API documentation
- [TypeScript SDK Guide](packages/sdk/docs/QUICKSTART.md) - Getting started with the SDK
- [MCP Server](docs/mcp.md) - Integration with AI coding assistants

## Citation

If you use Delta in your research, please cite:

```bibtex
@software{delta2026,
  title={Delta: Lossless Token Sequence Compression for Large Language Models},
  author={{Triage Sec}},
  year={2026},
  url={https://github.com/delta-ltsc/delta}
}
```

This work builds on foundational research in lossless token compression:

```bibtex
@article{harvill2024lossless,
  title={Lossless Token Sequence Compression via Meta-Tokens},
  author={Harvill, John and others},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass (`pytest` for Python, `npm test` for TypeScript)
2. Code is formatted (`ruff format` for Python, `prettier` for TypeScript)
3. Type hints are complete (`mypy delta/` for Python)
4. New features include tests and documentation

## Acknowledgments

- The foundational LTSC algorithm from Harvill et al. (2024)
- The open-source community for feedback and contributions
