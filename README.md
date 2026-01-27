# small

Lossless Token Sequence Compression (LTSC) reference implementation for internal use.

## Intent

Small targets efficient, lossless compression of token sequences before LLM ingestion. The system builds on "Lossless Token Sequence Compression via Meta-Tokens" and focuses on strict reversibility, predictable formatting, and measurable compression gains on structured inputs.

## Objectives

- Maintain perfect round-trip reconstruction.
- Achieve substantial compression on code and structured documents.
- Keep compression latency low for sequences up to 8192 tokens.
- Provide a format learnable by transformer models.

## Constraints

- Lossless property is non-negotiable.
- Dictionary format must be consistent and parseable.
- Core compression should target O(n log n) worst-case behavior.
- Compression must obey the strict compressibility inequality.

## Repository Layout

- `small/` Core library.
- `docs/` Design and format documentation.
- `tests/` Unit tests.
- `examples/` Minimal usage examples.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Example

```bash
python examples/basic.py
```

## Benchmark

```bash
python benchmarks/latency.py --tokens 8192 --runs 10
```

```bash
python benchmarks/ratio.py --tokens 8192 --runs 10
```

## Documentation

- `docs/00-intent.md`
- `docs/01-baseline.md`
- `docs/02-format.md`
- `docs/03-usage.md`
- `docs/04-verification.md`
