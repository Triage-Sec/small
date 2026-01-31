# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Delta is a research-grade lossless token sequence compression system for LLMs. It replaces repeated multi-token patterns with compact meta-token references, achieving compression on structured inputs while guaranteeing perfect reconstruction.

## Development Commands

### Python Core

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_compress.py -v

# Run with coverage
pytest --cov=delta --cov-report=html

# Format code
ruff format

# Lint
ruff check

# Type checking
mypy delta/
```

### Monorepo Packages

The `packages/` directory contains multi-language implementations:

```bash
# Build all npm packages (WASM core + SDK + ML)
./scripts/build-all.sh

# Build individual packages
./scripts/build-wasm.sh      # Rust WASM core (packages/core/)
./scripts/build-sdk.sh       # TypeScript SDK (packages/sdk/)
./scripts/build-ml.sh        # TypeScript ML utilities (packages/ml/)

# SDK tests
cd packages/sdk && npm test              # vitest
cd packages/sdk && npm run test:browser  # playwright
```

### Benchmarks

```bash
python benchmarks/ratio.py --tokens 8192 --runs 10
python benchmarks/latency.py --tokens 8192 --runs 10
python benchmarks/length_compare.py --tokens 8192 --lengths 2,3,4,5,6
```

## Architecture

### Compression Pipeline

```
Tokens → Discovery → Selection → Replacement → Serialization
```

1. **Discovery** (`delta/discovery*.py`): Finds repeated patterns
   - `discovery_sa.py`: Suffix array (default, O(n log n))
   - `discovery.py`: Sliding window (small inputs)
   - `bpe_discovery.py`: BPE-style merging (hierarchical)
   - `ast_python.py`: Python AST-aware
   - `fuzzy.py`: Near-duplicate clustering

2. **Selection** (`delta/selection*.py`): Chooses non-overlapping occurrences
   - `selection_mode="greedy"`: O(n log n), local optimal
   - `selection_mode="optimal"`: Weighted interval DP
   - `selection_mode="beam"`: Bounded exploration
   - `selection_mode="ilp"`: Global optimal via ILP solver

3. **Replacement** (`delta/swap.py`, `delta/dictionary.py`): Substitutes patterns with meta-tokens

4. **Serialization** (`delta/serialization.py`): Produces `<Dict>...</Dict>` format

### Key Entry Points

- `delta/compressor.py`: Public API (`compress()`, `decompress()`)
- `delta/engine.py`: Pipeline orchestration (`CompressionEngine`)
- `delta/config.py`: All configuration options (`CompressionConfig`)

### ML Integration

Optional features controlled via `CompressionConfig`:
- **Subsumption pruning** (`subsumption.py`): Removes redundant patterns
- **Importance scoring** (`pattern_importance.py`): Adjusts priorities based on semantic importance
- **Region detection** (`adaptive.py`): Different compression for SYSTEM/USER/CODE regions
- **Quality prediction** (`quality_predictor.py`): Validates compression before output

### Monorepo Structure

- `delta/`: Python core library
- `packages/core/`: Rust WASM implementation
- `packages/sdk/`: TypeScript SDK (wraps WASM)
- `packages/ml/`: TypeScript ML utilities for client-side quality prediction

## Testing Patterns

Test fixtures are in `tests/fixtures/`:
- `corpora/code/`: Python/TypeScript code samples
- `corpora/policies/`: Security policy documents
- `corpora/edge_cases/`: Minimal/maximal repetition cases
- `benchmarks/`: Benchmark input data

Use `conftest.py` fixtures for corpus loading:
```python
def test_something(python_delta_corpus):
    for doc in python_delta_corpus:
        # doc is a CorpusDocument
        pass
```

## Configuration

All settings flow through `CompressionConfig` (immutable dataclass). Key options:

- `discovery_mode`: "suffix-array" | "sliding-window" | "bpe"
- `selection_mode`: "greedy" | "optimal" | "beam" | "ilp"
- `hierarchical_enabled`: Multi-pass compression (up to `hierarchical_max_depth`)
- `enable_subsumption_pruning`: Remove redundant patterns
- `use_importance_scoring`: ML-based priority adjustment
- `verify`: Enable round-trip verification

## Compressibility Constraint

A pattern is only compressed if: `length × count > 1 + length + count + overhead`

This guarantees compression never increases sequence length.
