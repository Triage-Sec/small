"""Small: Lossless Token Sequence Compression (LTSC)."""

from .compressor import compress, decompress, CompressionConfig, CompressionResult

__all__ = ["compress", "decompress", "CompressionConfig", "CompressionResult"]
