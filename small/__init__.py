"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, decompress
from .config import CompressionConfig
from .types import CompressionResult

__all__ = ["compress", "decompress", "CompressionConfig", "CompressionResult"]
