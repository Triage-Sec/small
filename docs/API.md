# API Reference

Complete API documentation for Delta.

## Core Functions

### `compress`

```python
def compress(
    tokens: Sequence[Token],
    config: CompressionConfig | None = None,
) -> CompressionResult
```

Compress a token sequence using the configured strategy.

**Parameters:**
- `tokens`: Input token sequence (any hashable type)
- `config`: Compression configuration (uses defaults if None)

**Returns:** `CompressionResult` with compressed output and metadata

**Example:**
```python
from delta import compress, CompressionConfig

result = compress(["a", "b", "a", "b"], CompressionConfig())
print(result.compressed_length)
```

### `decompress`

```python
def decompress(
    tokens: Sequence[Token],
    config: CompressionConfig | None = None,
) -> list[Token]
```

Decompress a compressed token sequence back to original.

**Parameters:**
- `tokens`: Compressed token sequence (from `result.serialized_tokens`)
- `config`: Must match config used for compression

**Returns:** Original token sequence

**Raises:** `ValueError` if format is invalid

### `compress_python_source`

```python
def compress_python_source(
    source: str,
    config: CompressionConfig | None = None,
) -> tuple[list[Token], CompressionResult]
```

Compress Python source code with AST-aware pattern discovery.

**Parameters:**
- `source`: Python source code string
- `config`: Compression configuration

**Returns:** Tuple of (tokenized source, compression result)

## Configuration

### `CompressionConfig`

```python
@dataclass(frozen=True)
class CompressionConfig:
    # Pattern discovery
    min_subsequence_length: int = 2
    max_subsequence_length: int = 8
    discovery_mode: str = "suffix-array"  # "suffix-array", "sliding-window", "bpe"
    
    # Selection algorithm
    selection_mode: str = "greedy"  # "greedy", "optimal", "beam", "ilp", "semantic"
    beam_width: int = 8
    ilp_time_limit: float = 1.0
    
    # Semantic selection (requires embedding provider)
    semantic_context_window: int = 8
    semantic_similarity_threshold: float = 0.7
    semantic_diversity_penalty: float = 0.5
    semantic_embedding_provider: str | None = None  # "openai", "voyage", etc.
    semantic_embedding_model: str | None = None
    
    # Hierarchical compression
    hierarchical_enabled: bool = True
    hierarchical_max_depth: int = 3
    hierarchical_min_improvement: float = 0.02
    
    # Dictionary format
    dict_length_enabled: bool = True
    meta_token_prefix: str = "<MT_"
    meta_token_suffix: str = ">"
    meta_token_pool_size: int = 500
    
    # BPE discovery
    enable_bpe_discovery: bool = False
    bpe_max_iterations: int = 100
    
    # Subsumption
    enable_subsumption_pruning: bool = True
    subsumption_min_independent: int = 2
    
    # ML features
    use_importance_scoring: bool = False
    importance_scorer_type: str = "composite"
    importance_weight: float = 0.5
    enable_adaptive_regions: bool = False
    enable_quality_prediction: bool = False
    quality_task_type: str = "general"
    max_predicted_degradation: float = 0.05
    
    # Verification
    verify: bool = False
    metrics_enabled: bool = True
```

## Result Types

### `CompressionResult`

```python
@dataclass(frozen=True)
class CompressionResult:
    original_tokens: tuple[Token, ...]
    compressed_tokens: list[Token]
    serialized_tokens: list[Token]
    dictionary_tokens: list[Token]
    body_tokens: list[Token]
    dictionary_map: dict[Token, tuple[Token, ...]]
    meta_tokens_used: tuple[Token, ...]
    original_length: int
    compressed_length: int
    static_dictionary_id: str | None
    metrics: object | None
    dictionary: object | None
    
    def verify(self, original_tokens: Sequence[Token], config: CompressionConfig) -> None:
        """Verify lossless round-trip. Raises ValueError on failure."""
```

### `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    subsequence: tuple[Token, ...]  # The pattern
    length: int                      # len(subsequence)
    positions: tuple[int, ...]       # Non-overlapping positions
    priority: int = 0                # Selection priority
    patches: dict[int, tuple] = {}   # Fuzzy match patches
```

### `Occurrence`

```python
@dataclass(frozen=True)
class Occurrence:
    start: int                       # Start position
    length: int                      # Pattern length
    subsequence: tuple[Token, ...]   # The pattern
    priority: int = 0                # Selection priority
    patches: tuple = ()              # Fuzzy patches
```

## Discovery Functions

### `discover_bpe_candidates`

```python
def discover_bpe_candidates(
    tokens: Sequence[Token],
    config: CompressionConfig,
    max_iterations: int = 100,
) -> list[Candidate]
```

BPE-style iterative pattern discovery.

### `discover_extended_bpe_candidates`

```python
def discover_extended_bpe_candidates(
    tokens: Sequence[Token],
    config: CompressionConfig,
) -> list[Candidate]
```

BPE discovery with pattern extension to longer sequences.

## Selection Functions

### `select_occurrences`

```python
def select_occurrences(
    candidates: Iterable[Candidate],
    config: CompressionConfig,
    tokens: Sequence[Token] | None = None,  # Required for semantic mode
) -> SelectionResult
```

Select non-overlapping occurrences using configured strategy.

**Selection modes:**
- `greedy`: O(n log n), savings-density heuristic
- `optimal`: O(n²), weighted interval scheduling via DP
- `beam`: O(n × width), beam search with marginal savings
- `ilp`: Globally optimal via ILP solver (requires scipy)
- `semantic`: Embedding-based selection (requires provider)

**Returns:** `SelectionResult` with `selected: list[Occurrence]`

### `select_occurrences_semantic`

```python
def select_occurrences_semantic(
    candidates: Iterable[Candidate],
    tokens: Sequence[Token],
    config: CompressionConfig,
    provider: EmbeddingProvider,
) -> list[Occurrence]
```

Semantic-aware selection using embedding similarity. Patterns with similar contexts across occurrences are preferred (redundant information). Patterns with diverse contexts are deprioritized (unique meaning each time).

## Subsumption Functions

### `build_subsumption_graph`

```python
def build_subsumption_graph(candidates: list[Candidate]) -> SubsumptionGraph
```

Build directed graph of pattern subsumption relationships.

### `prune_subsumed_candidates`

```python
def prune_subsumed_candidates(
    candidates: list[Candidate],
    config: CompressionConfig | None = None,
    min_independent_occurrences: int = 2,
) -> list[Candidate]
```

Remove patterns fully subsumed by longer patterns.

### `deduplicate_candidates`

```python
def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]
```

Merge duplicate candidates from different discovery strategies.

## Pattern Importance

### `ImportanceScorer` Protocol

```python
class ImportanceScorer(Protocol):
    def score_patterns(
        self,
        tokens: Sequence[Token],
        candidates: list[Candidate],
    ) -> list[float]:
        """Return importance scores in [0, 1] for each candidate."""
```

### Scorer Classes

```python
class PositionalImportanceScorer:
    decay_rate: float = 2.0

class FrequencyImportanceScorer:
    pass

class LengthImportanceScorer:
    pass

class EmbeddingImportanceScorer:
    provider: EmbeddingProvider
    context_window: int = 5
    max_samples: int = 10

class CompositeImportanceScorer:
    scorers: list[tuple[ImportanceScorer, float]]
```

### `create_default_scorer`

```python
def create_default_scorer() -> CompositeImportanceScorer
```

Create composite scorer with default weights (positional 40%, frequency 30%, length 30%).

### `adjust_candidate_priorities`

```python
def adjust_candidate_priorities(
    candidates: list[Candidate],
    importance_scores: list[float],
    importance_weight: float = 0.5,
) -> list[Candidate]
```

Adjust priorities based on importance (low importance → higher compression priority).

## Adaptive Compression

### Region Types

```python
class RegionType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CONTEXT = "context"
    CODE = "code"
    DATA = "data"
    UNKNOWN = "unknown"
```

### `Region`

```python
@dataclass(frozen=True)
class Region:
    start: int
    end: int
    region_type: RegionType
    max_compression_ratio: float
    priority_boost: int
```

### `detect_regions`

```python
def detect_regions(
    tokens: Sequence[Token],
    markers: dict[str, RegionType] | None = None,
    default_type: RegionType = RegionType.UNKNOWN,
) -> list[Region]
```

Detect semantic regions based on marker tokens.

### `filter_candidates_by_region`

```python
def filter_candidates_by_region(
    candidates: list[Candidate],
    regions: list[Region],
    tokens: Sequence[Token],
) -> list[Candidate]
```

Apply region-based priority adjustments to candidates.

## Quality Prediction

### `QualityPrediction`

```python
@dataclass(frozen=True)
class QualityPrediction:
    predicted_degradation: float  # 0.0 to 1.0
    confidence: float
    recommendation: str           # "compress", "partial", "skip"
    suggested_max_ratio: float
    risk_factors: tuple[str, ...]
```

### `CompressionQualityPredictor`

```python
@dataclass
class CompressionQualityPredictor:
    model_path: str | None = None
    task_type: str = "general"
    conservative: bool = False
    
    def predict(
        self,
        tokens: Sequence[Token],
        proposed_result: CompressionResult,
    ) -> QualityPrediction
    
    def should_compress(
        self,
        tokens: Sequence[Token],
        proposed_result: CompressionResult,
        max_degradation: float = 0.05,
    ) -> bool
```

### `create_predictor`

```python
def create_predictor(task_type: str = "general") -> CompressionQualityPredictor
```

Factory function for quality predictors.

## Embedding Providers

### `EmbeddingProvider` Protocol

```python
class EmbeddingProvider(Protocol):
    def embed_single(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def dimension(self) -> int: ...
    def model_id(self) -> str: ...
```

### Provider Classes

```python
class HuggingFaceEmbeddingProvider:
    model_name: str
    device: str = "auto"
    batch_size: int = 32

class OpenAIEmbeddingProvider:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None

class OllamaEmbeddingProvider:
    model: str
    base_url: str = "http://localhost:11434"

class VoyageEmbeddingProvider:
    model: str
    api_key_env: str = "VOYAGE_API_KEY"

class CohereEmbeddingProvider:
    model: str
    api_key_env: str = "COHERE_API_KEY"
```

### `create_provider`

```python
def create_provider(
    name: str,
    model: str | None = None,
    **kwargs,
) -> EmbeddingProvider
```

Factory function to create embedding providers by name.

**Supported providers:**
- `"openai"` / `"gpt"`: OpenAI embeddings (requires `OPENAI_API_KEY`)
- `"voyage"` / `"voyageai"`: Voyage AI embeddings (requires `VOYAGE_API_KEY`)
- `"cohere"`: Cohere embeddings (requires `COHERE_API_KEY`)
- `"sentence-transformers"` / `"st"` / `"huggingface"`: Local sentence-transformers
- `"ollama"`: Local Ollama embeddings

**Example:**
```python
from delta.embeddings import create_provider

# OpenAI (uses OPENAI_API_KEY env var)
provider = create_provider("openai", model="text-embedding-3-small")

# Local sentence-transformers (no API key needed)
provider = create_provider("sentence-transformers", model="all-MiniLM-L6-v2")
```

## Training Utilities

### `TrainingExample`

```python
@dataclass
class TrainingExample:
    input_tokens: list[Token]
    target_tokens: list[Token]
    compression_ratio: float
```

### `generate_training_examples`

```python
def generate_training_examples(
    corpus: Iterable[Sequence[Token]],
    config: CompressionConfig,
    count: int,
) -> list[TrainingExample]
```

Generate training examples for fine-tuning.

### `build_curriculum`

```python
def build_curriculum(
    examples: list[TrainingExample],
    stages: int = 3,
) -> list[list[TrainingExample]]
```

Build curriculum learning stages (easy → hard).

## Static Dictionaries

### `build_static_dictionary`

```python
def build_static_dictionary(
    corpus: Iterable[Sequence[Token]],
    config: StaticDictionaryConfig,
) -> dict[Token, tuple[Token, ...]]
```

Build a static dictionary from a corpus for domain-specific compression.

### `save_static_dictionary` / `load_static_dictionary`

```python
def save_static_dictionary(path: str, dictionary: dict, metadata: dict | None = None) -> None
def load_static_dictionary(path: str) -> tuple[dict, dict]
```

Persist and load static dictionaries.

## Streaming Compression

### `StreamingCompressor`

```python
class StreamingCompressor:
    def __init__(
        self,
        config: CompressionConfig | None = None,
        chunk_size: int = 8192,
        overlap: int = 1024,
    )
    
    def add_chunk(self, tokens: Sequence[Token]) -> StreamingResult | None
    def finish(self) -> StreamingResult
    def stats(self) -> StreamingStats
```

Process arbitrarily large inputs with bounded memory.

### `compress_streaming`

```python
def compress_streaming(
    token_iterator: Iterable[Sequence[Token]],
    config: CompressionConfig | None = None,
) -> StreamingResult
```

Compress a stream of token chunks.

## Cross-Document Pattern Cache

### `PatternCache`

```python
class PatternCache:
    def __init__(
        self,
        max_patterns: int = 10000,
        bloom_filter_size: int = 100000,
    )
    
    def record_patterns(
        self,
        dictionary_map: dict[Token, tuple[Token, ...]],
        positions: dict[tuple[Token, ...], tuple[int, ...]],
    ) -> None
    
    def get_warm_start_candidates(
        self,
        tokens: Sequence[Token],
        top_k: int = 50,
    ) -> list[tuple[tuple[Token, ...], tuple[int, ...], int]]
    
    def might_contain(self, pattern: tuple[Token, ...]) -> bool
```

Learn and reuse patterns across compression operations.

## Template Extraction

### `Template`

```python
@dataclass(frozen=True)
class Template:
    frame: tuple[Token | SlotMarker, ...]  # Pattern with slot markers
    slot_count: int
    instances: tuple[TemplateInstance, ...]
```

### `discover_templates`

```python
def discover_templates(
    tokens: Sequence[Token],
    config: CompressionConfig,
) -> list[TemplateCandidate]
```

Discover parameterized patterns with variable slots.

## Quality Monitoring

### `QualityMonitor`

```python
class QualityMonitor:
    def __init__(self, config: MonitoringConfig | None = None)
    
    def record(
        self,
        result: CompressionResult,
        predicted_degradation: float = 0.0,
        latency_ms: float | None = None,
    ) -> QualityRecord
    
    def get_summary(self) -> QualitySummary | None
    def check_health(self) -> HealthStatus
    def learn_baseline(self, metric: str) -> QualityBaseline | None
```

Track compression quality over time with rolling statistics.

### `AlertManager`

```python
class AlertManager:
    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        cooldown_seconds: float = 300,
    )
    
    def check(self, summary: QualitySummary) -> list[QualityAlert]
    def subscribe(self, callback: Callable[[QualityAlert], None]) -> None
```

Configurable alerting on quality thresholds.

### Export Functions

```python
def export_prometheus(summary: QualitySummary, labels: dict | None = None) -> str
def export_summary_ascii(summary: QualitySummary) -> str
def export_health_ascii(monitor: QualityMonitor) -> str
```

Export monitoring data in various formats.

## Metrics

### `evaluate_compression_quality`

```python
def evaluate_compression_quality(
    tokens: Sequence[Token],
    result: CompressionResult,
) -> dict[str, float]
```

Compute quality metrics for a compression result.

**Returns:** Dict with keys:
- `compression_ratio`
- `dict_overhead_ratio`
- `avg_pattern_length`
- `effective_compression`
- `efficiency`
