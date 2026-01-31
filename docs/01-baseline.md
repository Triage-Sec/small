# Baseline Algorithm

## Compressibility Condition

A subsequence of length N appearing K non-overlapping times is compressible only when:

```
N * K > 1 + N + K
```

This yields exactly three allowed cases:

- N >= 4, K >= 2
- N = 3, K >= 3
- N = 2, K >= 4

Delta enforces this as a mandatory filter for all candidate patterns.

## Three-Phase Structure

1) **Discovery**
   - Enumerate candidate subsequences for lengths 2..max_len.
   - Count non-overlapping occurrences per subsequence (no shared token positions).
   - Filter candidates by the compressibility condition.
   - Process lengths from longest to shortest.

2) **Swapping**
   - Iterate candidates in descending length order.
   - Before swapping, ensure all candidate positions are still unconsumed.
   - Re-evaluate compressibility with remaining valid occurrences.
   - Replace valid occurrences with a meta-token.

3) **Dictionary Construction**
   - Build dictionary entries as meta-token + subsequence.
   - Wrap dictionary with <Dict> and </Dict> delimiters.
   - Prepend dictionary to the compressed body.

## Meta-Token Assignment

Meta-tokens are sampled uniformly from the available pool rather than assigned sequentially.
