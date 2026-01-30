# ML Integration Guide

Small provides several ML-powered features to improve compression quality and adapt to different use cases. This guide covers pattern importance scoring, region-aware compression, and quality prediction.

## Pattern Importance Scoring

Not all patterns are equally important for downstream tasks. Importance scoring helps preserve semantically critical patterns while aggressively compressing redundant content.

### Available Scorers

#### Positional Importance

Patterns at the beginning of a prompt (instructions, system context) are often more critical.

```python
from small.pattern_importance import PositionalImportanceScorer

scorer = PositionalImportanceScorer(decay_rate=2.0)
scores = scorer.score_patterns(tokens, candidates)
# scores[i] ∈ [0, 1], higher = more important (earlier)
```

#### Frequency Importance

Rare patterns may carry more unique information than common boilerplate.

```python
from small.pattern_importance import FrequencyImportanceScorer

scorer = FrequencyImportanceScorer()
scores = scorer.score_patterns(tokens, candidates)
# Inverse frequency: rare patterns get higher scores
```

#### Length Importance

Longer patterns often represent meaningful structures (function signatures, repeated blocks).

```python
from small.pattern_importance import LengthImportanceScorer

scorer = LengthImportanceScorer()
scores = scorer.score_patterns(tokens, candidates)
# Longer patterns get higher scores
```

#### Embedding-Based Importance

Patterns appearing in diverse semantic contexts carry unique information each time.

```python
from small.pattern_importance import EmbeddingImportanceScorer
from small.embeddings import HuggingFaceEmbeddingProvider

provider = HuggingFaceEmbeddingProvider(model_name="all-MiniLM-L6-v2")
scorer = EmbeddingImportanceScorer(
    provider=provider,
    context_window=5,
    max_samples=10,
)
scores = scorer.score_patterns(tokens, candidates)
```

**Intuition**: A pattern appearing in semantically similar contexts is redundant (compress aggressively). A pattern appearing in diverse contexts carries unique information (preserve carefully).

### Composite Scoring

Combine multiple signals with configurable weights:

```python
from small.pattern_importance import CompositeImportanceScorer, create_default_scorer

# Default: positional (40%), frequency (30%), length (30%)
scorer = create_default_scorer()

# Or custom weights
scorer = CompositeImportanceScorer(
    scorers=[
        (PositionalImportanceScorer(), 0.5),
        (FrequencyImportanceScorer(), 0.3),
        (LengthImportanceScorer(), 0.2),
    ]
)
```

### Applying Importance Scores

Adjust candidate priorities based on importance:

```python
from small.pattern_importance import adjust_candidate_priorities, filter_high_importance_candidates

# Adjust priorities (low importance → higher compression priority)
adjusted_candidates = adjust_candidate_priorities(
    candidates,
    importance_scores,
    importance_weight=0.5,
)

# Or filter out high-importance patterns entirely
filtered_candidates = filter_high_importance_candidates(
    candidates,
    importance_scores,
    threshold=0.9,  # Skip patterns with importance > 0.9
)
```

## Region-Aware Compression

Different parts of a prompt have different compression tolerances.

### Region Types

| Type | Description | Default Max Compression | Priority Boost |
|------|-------------|------------------------|----------------|
| `SYSTEM` | System instructions | 95% (minimal) | -2 (preserve) |
| `USER` | User input | 85% (moderate) | 0 |
| `ASSISTANT` | Assistant history | 80% | +1 |
| `CONTEXT` | Retrieved documents | 50% (aggressive) | +3 |
| `CODE` | Code blocks | 75% | +1 |
| `DATA` | Structured data (JSON) | 40% (very aggressive) | +4 |

### Region Detection

Automatic detection based on markers:

```python
from small.adaptive import detect_regions, RegionType

tokens = ["[SYSTEM]", "You", "are", "helpful", "[USER]", "Hello", "```", "code", "```"]
regions = detect_regions(tokens)

for region in regions:
    print(f"{region.region_type}: positions {region.start}-{region.end}")
```

**Default markers**:
- `[SYSTEM]`, `<<SYS>>` → SYSTEM
- `[USER]`, `[INST]` → USER
- `[CONTEXT]` → CONTEXT
- `` ``` `` → CODE
- `{` → DATA

### Custom Markers

```python
from small.adaptive import detect_regions, RegionType

custom_markers = {
    "<|system|>": RegionType.SYSTEM,
    "<|user|>": RegionType.USER,
    "<|assistant|>": RegionType.ASSISTANT,
}

regions = detect_regions(tokens, markers=custom_markers)
```

### Heuristic Detection

When no markers are present:

```python
from small.adaptive import detect_regions_heuristic

# Assume first 10% is system context
regions = detect_regions_heuristic(tokens, system_fraction=0.1)
```

### Applying Region Filters

```python
from small.adaptive import filter_candidates_by_region

# Adjust candidates based on their containing regions
filtered_candidates = filter_candidates_by_region(candidates, regions, tokens)
```

Candidates in SYSTEM regions get negative priority boosts (less likely to compress). Candidates in CONTEXT/DATA regions get positive boosts (more likely to compress).

## Quality Prediction

Before committing to compression, predict whether it will hurt downstream task performance.

### Basic Usage

```python
from small import compress, CompressionConfig
from small.quality_predictor import create_predictor

config = CompressionConfig()
result = compress(tokens, config)

predictor = create_predictor(task_type="general")
prediction = predictor.predict(tokens, result)

print(f"Predicted degradation: {prediction.predicted_degradation:.1%}")
print(f"Recommendation: {prediction.recommendation}")
print(f"Risk factors: {prediction.risk_factors}")
```

### Task Types

The predictor adjusts sensitivity based on task type:

| Task Type | Sensitivity | Use Case |
|-----------|-------------|----------|
| `general` | 1.0× | Default balanced |
| `code` | 1.5× | Code generation (sensitive) |
| `policy` | 1.3× | Legal/policy text |
| `math` | 1.4× | Mathematical content |
| `creative` | 0.8× | Creative writing (tolerant) |
| `summarization` | 0.9× | Summarization |
| `qa` | 1.2× | Question answering |
| `classification` | 0.7× | Classification (robust) |

### Recommendations

The predictor returns one of three recommendations:

- **`compress`**: Safe to use compressed output (degradation < 2%)
- **`partial`**: Use with caution, consider less aggressive settings (2-10%)
- **`skip`**: Skip compression for this input (degradation > 10%)

### Conservative Mode

For production systems where quality is critical:

```python
from small.quality_predictor import CompressionQualityPredictor

predictor = CompressionQualityPredictor(
    task_type="code",
    conservative=True,  # 30% higher degradation estimates
)
```

### Feature Extraction

The predictor uses these features:

- **Compression ratio**: `compressed_length / original_length`
- **Pattern statistics**: average length, count, variance
- **Dictionary overhead**: `dict_tokens / compressed_length`
- **Token diversity**: unique tokens before/after
- **Coverage ratio**: tokens replaced / total tokens

### Simple Decision API

```python
predictor = create_predictor("qa")

if predictor.should_compress(tokens, result, max_degradation=0.05):
    output = result.serialized_tokens
else:
    output = list(tokens)  # Use original
```

## Integration Example

Complete workflow combining all ML features:

```python
from small import compress, CompressionConfig
from small.discovery_sa import discover_candidates_sa
from small.pattern_importance import create_default_scorer, adjust_candidate_priorities
from small.adaptive import detect_regions, filter_candidates_by_region
from small.subsumption import prune_subsumed_candidates
from small.quality_predictor import create_predictor

# 1. Discover candidates
config = CompressionConfig()
candidates = discover_candidates_sa(tokens, config)

# 2. Prune subsumed patterns
candidates = prune_subsumed_candidates(candidates, config)

# 3. Detect regions and filter
regions = detect_regions(tokens)
candidates = filter_candidates_by_region(candidates, regions, tokens)

# 4. Score importance and adjust priorities
scorer = create_default_scorer()
scores = scorer.score_patterns(tokens, candidates)
candidates = adjust_candidate_priorities(candidates, scores)

# 5. Compress with adjusted candidates
result = compress(tokens, config)  # Uses standard pipeline

# 6. Predict quality and decide
predictor = create_predictor("qa")
prediction = predictor.predict(tokens, result)

if prediction.recommendation == "compress":
    final_output = result.serialized_tokens
elif prediction.recommendation == "partial":
    # Re-compress with more conservative settings
    conservative_config = CompressionConfig(max_subsequence_length=4)
    result = compress(tokens, conservative_config)
    final_output = result.serialized_tokens
else:
    final_output = list(tokens)
```

## Semantic Selection

Semantic selection uses embeddings to make smarter pattern selection decisions. Patterns appearing in semantically similar contexts are better compression candidates (redundant information), while patterns in diverse contexts carry unique meaning and should be preserved.

### Basic Usage

```python
from small import compress, CompressionConfig

config = CompressionConfig(
    selection_mode="semantic",
    semantic_embedding_provider="openai",  # or "voyage", "sentence-transformers"
    semantic_embedding_model="text-embedding-3-small",
)

result = compress(tokens, config)
```

### How It Works

1. **Context extraction**: For each pattern occurrence, extract surrounding tokens as context
2. **Embedding**: Get embeddings for all context windows via the configured provider
3. **Similarity scoring**: Compute average pairwise cosine similarity across occurrences
4. **Weight adjustment**:
   - High similarity (above threshold) → boost pattern weight (good to compress)
   - Low similarity (diverse contexts) → reduce pattern weight (preserve unique meaning)
5. **Selection**: Run weighted interval scheduling with adjusted weights

### Configuration Options

```python
config = CompressionConfig(
    selection_mode="semantic",
    
    # Provider configuration
    semantic_embedding_provider="openai",  # Required for semantic mode
    semantic_embedding_model="text-embedding-3-small",  # Provider-specific model
    
    # Algorithm parameters
    semantic_context_window=8,           # Tokens around occurrence for context
    semantic_similarity_threshold=0.7,   # Similarity above this = good candidate
    semantic_diversity_penalty=0.5,      # How much to penalize diverse patterns
)
```

### Supported Embedding Providers

| Provider | Env Variable | Default Model | Notes |
|----------|--------------|---------------|-------|
| `openai` | `OPENAI_API_KEY` | text-embedding-3-small | Fast, cheap |
| `voyage` | `VOYAGE_API_KEY` | voyage-3-lite | High quality |
| `cohere` | `COHERE_API_KEY` | embed-english-v3.0 | Good for English |
| `sentence-transformers` | (none) | all-MiniLM-L6-v2 | Local, no API |
| `ollama` | (none) | nomic-embed-text | Local, configurable |

### Fallback Behavior

If semantic selection is requested but no provider is available (missing API key or package), it gracefully falls back to `optimal` selection with a warning:

```python
# No OPENAI_API_KEY set
config = CompressionConfig(
    selection_mode="semantic",
    semantic_embedding_provider="openai",
)

# Warning: selection_mode='semantic' requires semantic_embedding_provider...
# Falls back to 'optimal' selection
result = compress(tokens, config)
```

### Direct API Usage

For more control, use the semantic selection function directly:

```python
from small.embeddings import create_provider
from small.selection_semantic import select_occurrences_semantic
from small.discovery_sa import discover_candidates_sa

# Create provider
provider = create_provider("openai", model="text-embedding-3-small")

# Discover candidates
candidates = discover_candidates_sa(tokens, config)

# Select with semantic weights
selected = select_occurrences_semantic(candidates, tokens, config, provider)
```

## Configuration Reference

Enable ML features via `CompressionConfig`:

```python
config = CompressionConfig(
    # Importance scoring
    use_importance_scoring=True,
    importance_scorer_type="composite",  # "positional", "frequency", "embedding"
    importance_weight=0.5,
    importance_filter_threshold=0.9,
    
    # Adaptive regions
    enable_adaptive_regions=True,
    region_markers=None,  # Use defaults, or provide custom dict
    adaptive_system_fraction=0.1,
    
    # Quality prediction
    enable_quality_prediction=True,
    quality_task_type="general",
    max_predicted_degradation=0.05,
    quality_conservative=False,
    
    # Semantic selection
    selection_mode="semantic",
    semantic_embedding_provider="openai",
    semantic_embedding_model="text-embedding-3-small",
    semantic_context_window=8,
    semantic_similarity_threshold=0.7,
    semantic_diversity_penalty=0.5,
)
```
