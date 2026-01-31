# Improvements and Extensions

This document summarizes the major improvements and research-grade extensions available in Delta, organized by impact and scope.

## Critical Algorithmic Fixes

### Selection Correctness

- Compressibility checks are now integrated into selection rather than post-filtering, preventing wasted selection capacity.
- Weighted selection uses marginal savings that account for dictionary overhead.
- Greedy selection uses a savings-density heuristic to prefer higher-value patterns.

### Hierarchical Early Stopping

- Hierarchical compression stops early when improvement falls below a threshold.
- Prevents unnecessary passes and reduces latency on diminishing returns.

## Pattern Discovery Improvements

### Suffix Array Optimization

- Maximal repeat extraction avoids generating all lengths for each LCP interval.
- Reduces candidate explosion on highly repetitive inputs.

### BPE-Style Iterative Discovery

- Iteratively merges the most beneficial adjacent pairs.
- Captures hierarchical patterns that one-shot discovery can miss.
- Useful for structured inputs and code repetition.

## Selection Enhancements

### ILP-Based Selection (Optional)

- Integer Linear Programming finds globally optimal non-overlapping patterns.
- Enforces compressibility constraints directly in the optimizer.
- Falls back to beam search if `scipy` is unavailable or timeouts occur.

### Beam Search with Marginal Savings

- Beam search evaluates marginal gains to balance exploration and efficiency.
- More stable than naive greedy on overlapping high-value candidates.

## Subsumption Analysis

- Detects and removes patterns fully subsumed by longer patterns.
- Keeps shorter patterns only if they have independent occurrences.
- Reduces dictionary size and redundancy.

## ML Integration Features

### Pattern Importance Scoring

- Scores candidates by positional, frequency, length, or embedding-based signals.
- Adjusts compression priority to preserve semantically critical regions.

### Region-Aware Compression

- Detects semantic regions (SYSTEM, USER, CONTEXT, CODE, DATA).
- Applies different compression priorities per region.

### Quality Prediction

- Predicts downstream impact of compression using heuristic signals.
- Returns a recommendation: `compress`, `partial`, or `skip`.

## Performance Optimizations

### Parallel Discovery (Process-Based)

- ProcessPoolExecutor for true parallelism on CPU-bound discovery.
- Chunk overlap handling ensures correctness across boundaries.

### NumPy Suffix Arrays

- `suffix_array_fast.py` uses numpy for faster suffix array construction.
- 2–5x speedup on large inputs (> 1K tokens).

## Compatibility and Stability

- All improvements preserve strict losslessness.
- Backwards compatible with existing serialized format.
- Configurable feature flags allow safe incremental adoption.
