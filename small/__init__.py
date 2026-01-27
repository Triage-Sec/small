"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, compress_python_source, decompress
from .config import CompressionConfig
from .engine import CompressionEngine, default_engine
from .sequence import TokenSequence
from .training import TrainingExample, build_example, build_curriculum, generate_training_examples
from .vocab import VocabExtension, plan_vocab_extension
from .types import CompressionResult

__all__ = [
    "compress",
    "compress_python_source",
    "decompress",
    "CompressionConfig",
    "CompressionResult",
    "CompressionEngine",
    "TokenSequence",
    "default_engine",
    "TrainingExample",
    "build_example",
    "build_curriculum",
    "generate_training_examples",
    "VocabExtension",
    "plan_vocab_extension",
]
