"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, compress_python_source, decompress, decompress_with_dictionary
from .config import CompressionConfig
from .engine import CompressionEngine, default_engine
from .sequence import TokenSequence
from .serialization import SerializedOutput, serialize
from .training import TrainingExample, build_example, build_curriculum, generate_training_examples
from .vocab import VocabExtension, plan_vocab_extension
from .types import CompressionResult
from .corpus import CorpusDocument, load_directory, load_jsonl
from .preprocess import PreprocessConfig, preprocess_corpus
from .analysis import AnalysisConfig, compute_document_weights
from .static_dict_builder import StaticDictionaryConfig, build_static_dictionary
from .static_dict_io import load_static_dictionary, save_static_dictionary
from .offline_pipeline import OfflinePipelineConfig, run_offline_analysis
from .metrics_writer import write_cache_stats_jsonl, write_metrics_jsonl
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
    "compress",
    "compress_python_source",
    "decompress",
    "decompress_with_dictionary",
    "CompressionConfig",
    "CompressionResult",
    "CompressionEngine",
    "TokenSequence",
    "default_engine",
    "SerializedOutput",
    "serialize",
    "TrainingExample",
    "build_example",
    "build_curriculum",
    "generate_training_examples",
    "VocabExtension",
    "plan_vocab_extension",
    "CorpusDocument",
    "load_directory",
    "load_jsonl",
    "PreprocessConfig",
    "preprocess_corpus",
    "AnalysisConfig",
    "compute_document_weights",
    "StaticDictionaryConfig",
    "build_static_dictionary",
    "save_static_dictionary",
    "load_static_dictionary",
    "OfflinePipelineConfig",
    "run_offline_analysis",
    "write_cache_stats_jsonl",
    "write_metrics_jsonl",
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
