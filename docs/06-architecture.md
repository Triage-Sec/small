# System Architecture

## Pipeline Stages

Small is structured as a pipeline of independent stages:

1) Tokenization (external adapter)
2) Pattern discovery
3) Pattern selection
4) Replacement and dictionary construction
5) Verification (optional)

Each stage has explicit input/output contracts and can be swapped or composed.

## Tokenizer Interface

`small.tokenizer` defines adapters that support:

- `encode(text) -> tokens`
- `decode(tokens) -> text`
- `vocab_size()`
- `is_special_token(token)`

Adapters include Hugging Face, tiktoken, and SentencePiece.

## Pattern Discovery

Discovery stages return candidates with subsequences and positions. Implementations include:

- Exact discovery (suffix array or sliding window)
- Fuzzy discovery (near-duplicate clustering + patches)
- AST discovery (Python)

## Pattern Selection

Selection strategies are configurable via `selection_mode`:

- `greedy`
- `optimal` (weighted interval scheduling)
- `beam`

## Compression Dictionary

`CompressionDictionary` provides:

- Meta-token to subsequence lookup
- Subsequence to meta-token lookup
- Hierarchical ordering
- Size limits

## Efficient Substring Search

Suffix arrays and LCP arrays are used for discovery:

- `build_suffix_array` uses the doubling algorithm.
- `lcp_intervals` enumerates repeated substring ranges.
- For hierarchical passes, suffix arrays are rebuilt on the new sequence.

## Configuration Defaults

Key defaults:

- min subsequence length: 2
- max subsequence length: 8
- max meta-tokens: 500
- max hierarchical depth: 3
- selection mode: greedy

Configuration validation raises on invalid settings and emits warnings for suboptimal choices.
