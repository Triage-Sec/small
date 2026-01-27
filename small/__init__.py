"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, compress_python_source, decompress
from .config import CompressionConfig
from .types import CompressionResult

__all__ = ["compress", "compress_python_source", "decompress", "CompressionConfig", "CompressionResult"]
