# Usage

Delta operates on token sequences, not raw text. Tokens are represented as hashable values (typically strings or integers). You can integrate Delta with any tokenizer by converting text to tokens before compression.

## High-Level Flow

1) Tokenize input with your model's tokenizer  
2) Compress the token sequence  
3) Feed compressed prompt to the model  
4) Keep the answer uncompressed during fine-tuning  
5) Compute loss only on answer tokens during fine-tuning  

## Minimal Example

```python
from delta import compress, decompress, CompressionConfig

tokens = ["the", "quick", "brown", "fox"] * 10
config = CompressionConfig(verify=True)

result = compress(tokens, config)
restored = decompress(result.serialized_tokens, config)

assert restored == tokens
```

## Configuration Guide

```python
from delta import CompressionConfig

config = CompressionConfig(
    # Discovery
    min_subsequence_length=2,
    max_subsequence_length=8,
    discovery_mode="suffix-array",  # or "sliding-window", "bpe"

    # Selection
    selection_mode="greedy",        # or "optimal", "beam", "ilp"
    beam_width=8,

    # Hierarchical compression
    hierarchical_enabled=True,
    hierarchical_max_depth=3,

    # Verification
    verify=True,
)
```

## Selection Modes

- `greedy`: Fast savings-density heuristic (default)
- `optimal`: Weighted interval scheduling (stronger compression, slower)
- `beam`: Beam search balancing quality vs. speed
- `ilp`: Globally optimal (requires `scipy`)

## Discovery Modes

- `suffix-array` (default): Best overall for structured inputs
- `sliding-window`: Useful for small inputs
- `bpe`: Iterative pair merging for hierarchical patterns

## Advanced Features

### AST-Aware Compression (Python)

```python
from delta import compress_python_source

tokens, result = compress_python_source("def foo(): return bar()")
```

### Region-Aware Compression

```python
from delta import detect_regions, filter_candidates_by_region

regions = detect_regions(tokens)
filtered = filter_candidates_by_region(candidates, regions, tokens)
```

### Importance Scoring

```python
from delta import create_default_scorer, adjust_candidate_priorities

scorer = create_default_scorer()
scores = scorer.score_patterns(tokens, candidates)
adjusted = adjust_candidate_priorities(candidates, scores)
```

### Quality Prediction

```python
from delta import create_predictor

predictor = create_predictor(task_type="code")
prediction = predictor.predict(tokens, result)
```

## Safety Notes

- Compression is lossless and reversible by design.
- Ensure `<Dict>` and `</Dict>` do not appear in original tokens.
- Configure `meta_token_prefix` to avoid collisions with your tokenizer.
- Use `verify=True` in development to catch regressions.
- Only enable `fuzzy_enabled` when patch encoding is acceptable.
