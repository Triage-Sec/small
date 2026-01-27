# Part 3 Improvements

## Hierarchical Compression

- Compression runs in multiple passes, allowing meta-tokens to reference previously defined meta-tokens.
- Dictionary entries include explicit length tokens to disambiguate nested references.
- Dictionary ordering is topologically sorted so dependencies appear before dependent entries.

## Grammar-Aware Compression (Python)

- Python source can be parsed into an AST to identify structurally identical subtrees.
- Subtree hashes normalize identifiers and literals, focusing on structure.
- Token ranges for repeated subtrees are prioritized in compression.
- Entry point: `compress_python_source`.

## Optimal Subsequence Selection

- `selection_mode` controls how non-overlapping occurrences are chosen:
  - `greedy`: fast baseline.
  - `optimal`: weighted interval scheduling for maximal savings.
  - `beam`: beam search for a middle ground.

## Static Domain Dictionaries

- Static dictionaries capture common patterns and avoid per-prompt dictionary overhead.
- Compressed output starts with a static dictionary marker token when used.
- Auto-detection uses conservative heuristics; it only applies a static dictionary above a confidence threshold.

## Fuzzy Matching (Minimal)

- Fuzzy matching groups near-duplicate subsequences by signature and Hamming distance.
- A canonical subsequence is stored in the dictionary; occurrences emit patches for differences.
- Patch encoding uses `<Patch>` delimiters and indexed replacement tokens.
