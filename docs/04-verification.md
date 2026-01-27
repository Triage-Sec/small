# Verification

Losslessness is enforced by round-trip checks during compression when enabled.

Verification includes:

- Decompression of the compressed sequence.
- Exact token-by-token equality with the original input.

Verification must be enabled in any pipeline where correctness is required.
