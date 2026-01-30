"""Small: Lossless Token Sequence Compression for Large Language Models.

Small is a research-grade lossless token sequence compression system designed
to reduce the computational and economic cost of LLM inference by eliminating
redundancy in input sequences before they reach the model.

Example:
    >>> from small import compress, decompress, CompressionConfig
    >>> tokens = ["the", "cat", "sat", "on", "the", "mat"] * 10
    >>> result = compress(tokens, CompressionConfig())
    >>> assert decompress(result.serialized_tokens) == tokens
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("small-ltsc")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for editable installs without metadata or direct source usage.
    __version__ = "0.0.0"

# Core compression API
from .compressor import (
    compress,
    compress_python_source,
    decompress,
    decompress_with_dictionary,
)
from .config import CompressionConfig
from .engine import CompressionEngine, default_engine
from .types import CompressionResult, Candidate, Occurrence

# Serialization
from .sequence import TokenSequence
from .serialization import SerializedOutput, serialize

# Pattern discovery
from .bpe_discovery import discover_bpe_candidates, discover_extended_bpe_candidates
from .subsumption import (
    build_subsumption_graph,
    prune_subsumed_candidates,
    deduplicate_candidates,
    SubsumptionGraph,
)

# Selection algorithms
from .selection import select_occurrences, SelectionResult

# ML integration
from .pattern_importance import (
    ImportanceScorer,
    EmbeddingImportanceScorer,
    PositionalImportanceScorer,
    CompositeImportanceScorer,
    create_default_scorer,
    adjust_candidate_priorities,
)
from .adaptive import (
    RegionType,
    Region,
    detect_regions,
    filter_candidates_by_region,
    AdaptiveCompressionConfig,
)
from .quality_predictor import (
    QualityPrediction,
    CompressionQualityPredictor,
    create_predictor,
    evaluate_compression_quality,
)

# Training utilities
from .training import (
    TrainingExample,
    build_example,
    build_curriculum,
    generate_training_examples,
)
from .vocab import VocabExtension, plan_vocab_extension

# Corpus management
from .corpus import CorpusDocument, load_directory, load_jsonl
from .preprocess import PreprocessConfig, preprocess_corpus
from .analysis import AnalysisConfig, compute_document_weights

# Static dictionaries
from .static_dict_builder import StaticDictionaryConfig, build_static_dictionary
from .static_dict_io import load_static_dictionary, save_static_dictionary

# Offline analysis pipeline
from .offline_pipeline import OfflinePipelineConfig, run_offline_analysis

# Metrics
from .metrics_writer import (
    write_cache_stats_jsonl,
    write_metrics_jsonl,
    write_offline_metrics_jsonl,
    write_combined_metrics_jsonl,
)
from .config_helpers import with_cache_stats_source

# Embedding providers
from .embeddings import (
    CohereEmbeddingProvider,
    EmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
)
from .embedding_cache import (
    RedisCacheConfig,
    RedisEmbeddingCache,
    SQLiteCacheConfig,
    SQLiteEmbeddingCache,
    cache_key,
)

__all__ = [
    # Version
    "__version__",
    # Core API
    "compress",
    "compress_python_source",
    "decompress",
    "decompress_with_dictionary",
    "CompressionConfig",
    "CompressionResult",
    "CompressionEngine",
    "default_engine",
    "Candidate",
    "Occurrence",
    # Serialization
    "TokenSequence",
    "SerializedOutput",
    "serialize",
    # Discovery
    "discover_bpe_candidates",
    "discover_extended_bpe_candidates",
    # Subsumption
    "build_subsumption_graph",
    "prune_subsumed_candidates",
    "deduplicate_candidates",
    "SubsumptionGraph",
    # Selection
    "select_occurrences",
    "SelectionResult",
    # ML Integration
    "ImportanceScorer",
    "EmbeddingImportanceScorer",
    "PositionalImportanceScorer",
    "CompositeImportanceScorer",
    "create_default_scorer",
    "adjust_candidate_priorities",
    "RegionType",
    "Region",
    "detect_regions",
    "filter_candidates_by_region",
    "AdaptiveCompressionConfig",
    "QualityPrediction",
    "CompressionQualityPredictor",
    "create_predictor",
    "evaluate_compression_quality",
    # Training
    "TrainingExample",
    "build_example",
    "build_curriculum",
    "generate_training_examples",
    "VocabExtension",
    "plan_vocab_extension",
    # Corpus
    "CorpusDocument",
    "load_directory",
    "load_jsonl",
    "PreprocessConfig",
    "preprocess_corpus",
    "AnalysisConfig",
    "compute_document_weights",
    # Static dictionaries
    "StaticDictionaryConfig",
    "build_static_dictionary",
    "save_static_dictionary",
    "load_static_dictionary",
    # Offline pipeline
    "OfflinePipelineConfig",
    "run_offline_analysis",
    # Metrics
    "write_cache_stats_jsonl",
    "write_metrics_jsonl",
    "write_offline_metrics_jsonl",
    "write_combined_metrics_jsonl",
    "with_cache_stats_source",
    # Embeddings
    "EmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "CohereEmbeddingProvider",
    "SQLiteCacheConfig",
    "SQLiteEmbeddingCache",
    "RedisCacheConfig",
    "RedisEmbeddingCache",
    "cache_key",
]
