"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, compress_python_source, decompress, decompress_with_dictionary
from .config import CompressionConfig
from .engine import CompressionEngine, default_engine
from .sequence import TokenSequence
from .serialization import SerializedOutput, serialize
from .training import TrainingExample, build_example, build_curriculum, generate_training_examples
from .vocab import VocabExtension, plan_vocab_extension
from .types import CompressionResult
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
