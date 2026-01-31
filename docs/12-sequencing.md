# Implementation Sequencing

## Phase One: Core Baseline

Implemented:

- Immutable token sequences
- Suffix array + LCP (doubling algorithm)
- Exact discovery and greedy selection
- Token replacement + dictionary construction
- Decompression + verification

Relevant modules:

- `delta/sequence.py`
- `delta/suffix_array.py`
- `delta/discovery_sa.py`
- `delta/selection.py`
- `delta/compressor.py`

## Phase Two: Selection Optimization

Implemented:

- Weighted interval scheduling (optimal)
- Beam search

Relevant modules:

- `delta/selection.py`
- `delta/config.py`

## Phase Three: Hierarchical Compression

Implemented:

- Multi-pass compression
- Length tokens for hierarchical parsing
- Topological dictionary ordering
- Recursive expansion

Relevant modules:

- `delta/compressor.py`
- `delta/dictionary.py`
- `delta/config.py`

## Phase Four: Grammar-Aware Compression

Implemented (Python):

- AST parsing and subtree hashing
- AST-derived candidate discovery
- Priority integration into selection

Relevant modules:

- `delta/ast_python.py`
- `delta/compressor.py`

## Phase Five: Domain Dictionaries

Implemented:

- Static dictionary registry and auto-detection
- Static marker format
- Apply static dictionary before dynamic compression
- Training utilities support

Relevant modules:

- `delta/static_dicts.py`
- `delta/domain.py`
- `delta/compressor.py`
- `delta/training.py`

## Phase Six: Fuzzy Matching

Implemented (minimal):

- Near-duplicate grouping with signatures + Hamming distance
- Canonical selection + patch encoding
- Lossless reconstruction

Relevant modules:

- `delta/fuzzy.py`
- `delta/dictionary.py`
- `delta/compressor.py`
