# Correctness and Verification

## Lossless Invariant

- Compression must be perfectly reversible.
- `decompress` reconstructs the original sequence by expanding meta-tokens.
- Hierarchical expansions are safe because dictionary dependencies are acyclic.

## Verification

- `CompressionResult.verify(original_tokens, config)` performs a round trip and asserts equality.
- Use `CompressionConfig.verify=True` during development and tests.
- In production, verification is optional but available for diagnostics.

## Property-Based Testing

The test suite includes randomized checks that validate:

- `decompress(compress(x)) == x`
- Compressed length is never greater than original length
- Dictionary consistency and non-overlapping selections
- Edge cases: empty inputs, short sequences, fully repeated sequences

## Metrics

Metrics are logged through `small.metrics`:

- Compression amount and ratio
- Effective savings
- Patterns discovered and used
- Average pattern length and frequency
- Dictionary overhead percentage
- Hierarchical depth utilization
