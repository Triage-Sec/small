# System Architecture

## Overview

Delta is structured as a modular pipeline where each stage has explicit input/output contracts and can be independently configured or replaced.

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐    ┌─────────────┐    ┌────────────────┐
│ Tokenization│ -> │   Discovery   │ -> │   Selection    │ -> │ Replacement │ -> │  Serialization │
│  (external) │    │ (candidates)  │    │ (occurrences)  │    │ (dictionary)│    │    (output)    │
└─────────────┘    └───────────────┘    └────────────────┘    └─────────────┘    └────────────────┘
                           │                    │
                           ▼                    ▼
                   ┌───────────────┐    ┌────────────────┐
                   │  Subsumption  │    │   Importance   │
                   │   Analysis    │    │    Scoring     │
                   └───────────────┘    └────────────────┘
```

## Pipeline Stages

### 1. Tokenization

External tokenizers convert text to token sequences. Delta operates on any hashable token type.

**Adapters** (`delta/tokenizer.py`):
- Hugging Face Transformers
- tiktoken (OpenAI)
- SentencePiece

### 2. Pattern Discovery

Discovery stages produce `Candidate` objects with subsequences and positions.

**Implementations**:

| Module | Strategy | Complexity | Best For |
|--------|----------|------------|----------|
| `discovery_sa.py` | Suffix array + LCP | O(n log n) | General use |
| `discovery.py` | Sliding window | O(n × L) | Small inputs |
| `bpe_discovery.py` | BPE-style merging | O(n × iterations) | Hierarchical patterns |
| `ast_python.py` | AST analysis | O(n) | Python code |
| `fuzzy.py` | Near-duplicate clustering | O(n²) | Noisy inputs |

### 3. Pattern Selection

Selection strategies choose non-overlapping occurrences that maximize compression.

**Implementations** (`selection.py`, `selection_ilp.py`):

| Mode | Algorithm | Complexity | Guarantee |
|------|-----------|------------|-----------|
| `greedy` | Savings-density sort | O(n log n) | Local optimal |
| `optimal` | Weighted interval DP | O(n log n) | Optimal for fixed patterns |
| `beam` | Beam search | O(n × width) | Bounded exploration |
| `ilp` | Integer LP | Exponential | Global optimal |

### 4. Replacement

Builds the dictionary and substitutes patterns with meta-tokens.

**Components**:
- `swap.py`: Performs token substitutions
- `dictionary.py`: Manages dictionary construction
- `dictionary_store.py`: Hierarchical ordering

### 5. Serialization

Produces the final compressed output format.

**Output structure**:
```
[static_marker?] <Dict> [entries...] </Dict> [body...]
```

## Engine Architecture

The `CompressionEngine` orchestrates the pipeline:

```python
@dataclass(frozen=True)
class CompressionEngine:
    discovery_stages: tuple[DiscoveryStage, ...]
    min_improvement_ratio: float = 0.02
    
    def compress_tokens(self, tokens, config) -> (body, dictionary):
        for depth in range(max_depth):
            candidates = self.discover(tokens, config)
            if not candidates:
                break
            
            selected = select_occurrences(candidates, config)
            tokens = apply_substitutions(tokens, selected)
            
            if improvement < min_improvement_ratio:
                break
        
        return tokens, dictionary
```

## ML Integration Points

### Subsumption Analysis

Removes redundant patterns before selection:

```
candidates → subsumption_graph → pruned_candidates → selection
```

### Importance Scoring

Adjusts selection priorities based on semantic importance:

```
candidates → importance_scorer → adjusted_priorities → selection
```

### Region-Aware Compression

Applies different strategies to different input regions:

```
tokens → detect_regions → filter_candidates_by_region → selection
```

### Quality Prediction

Validates compression before output:

```
result → quality_predictor → decision (compress/skip)
```

## Configuration System

`CompressionConfig` is an immutable dataclass with validation:

```python
config = CompressionConfig(
    # Core settings
    min_subsequence_length=2,
    max_subsequence_length=8,
    
    # Algorithm selection
    discovery_mode="suffix-array",
    selection_mode="greedy",
    
    # Features
    hierarchical_enabled=True,
    enable_subsumption_pruning=True,
    use_importance_scoring=False,
)
```

Validation raises on invalid settings:
- `min_subsequence_length < 2`
- `max_subsequence_length < min_subsequence_length`
- Unknown selection mode

## Data Flow

### Compression Flow

```
Input tokens
     │
     ▼
┌────────────────────────────────────────┐
│           Discovery Stage              │
│  suffix_array → lcp_intervals → filter │
└────────────────┬───────────────────────┘
                 │ list[Candidate]
                 ▼
┌────────────────────────────────────────┐
│         Subsumption Pruning            │
│   build_graph → prune_subsumed         │
└────────────────┬───────────────────────┘
                 │ list[Candidate]
                 ▼
┌────────────────────────────────────────┐
│          Selection Stage               │
│   build_occurrences → select_algo      │
└────────────────┬───────────────────────┘
                 │ list[Occurrence]
                 ▼
┌────────────────────────────────────────┐
│          Replacement Stage             │
│   build_dictionary → substitute        │
└────────────────┬───────────────────────┘
                 │ (body, dictionary)
                 ▼
┌────────────────────────────────────────┐
│         Serialization Stage            │
│   format_dict → concat_body            │
└────────────────┬───────────────────────┘
                 │
                 ▼
Output: serialized_tokens
```

### Decompression Flow

```
Compressed tokens
     │
     ▼
┌────────────────────────────────────────┐
│           Parse Dictionary             │
│  find_delimiters → extract_entries     │
└────────────────┬───────────────────────┘
                 │ dict[meta → sequence]
                 ▼
┌────────────────────────────────────────┐
│            Expand Body                 │
│   for token in body: expand(token)     │
└────────────────┬───────────────────────┘
                 │
                 ▼
Output: original_tokens
```

## Extension Points

### Custom Discovery

```python
@dataclass(frozen=True)
class CustomDiscoveryStage(DiscoveryStage):
    def discover(self, tokens, config) -> list[Candidate]:
        # Custom implementation
        return candidates
```

### Custom Selection

```python
if config.selection_mode == "custom":
    from mymodule import custom_select
    selected = custom_select(occurrences, config)
```

### Custom Importance Scorer

```python
class MyScorer:
    def score_patterns(self, tokens, candidates) -> list[float]:
        return [my_score(c) for c in candidates]
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Suffix array build | O(n log n) |
| LCP computation | O(n) |
| Greedy selection | O(n log n) |
| Hierarchical (d passes) | O(d × n log n) |

### Space Complexity

| Structure | Size |
|-----------|------|
| Suffix array | O(n) |
| LCP array | O(n) |
| Candidate list | O(k) where k = unique patterns |
| Dictionary | O(m × L) where m = selected patterns |

### Optimization Notes

- Suffix array uses numpy for inputs > 1000 tokens
- Parallel discovery uses ProcessPoolExecutor for inputs > 5000 tokens
- Early stopping triggers when improvement < 2% per pass
- ILP solver times out after 1 second by default
