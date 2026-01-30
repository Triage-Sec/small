"""Streaming compression with bounded memory.

This module provides incremental compression for large inputs that cannot
fit entirely in memory. Data is processed in chunks with configurable
overlap to detect cross-chunk patterns.

Example:
    >>> from small.streaming import StreamingCompressor
    >>> compressor = StreamingCompressor(chunk_size=4096)
    >>> for batch in token_batches:
    ...     result = compressor.add_tokens(batch)
    ...     if result:
    ...         process_compressed_chunk(result)
    >>> final = compressor.flush()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from .compressor import compress
from .config import CompressionConfig
from .types import Token

if TYPE_CHECKING:
    from .pattern_cache import PatternCache

logger = logging.getLogger(__name__)


@dataclass
class StreamingStats:
    """Cumulative statistics for streaming compression."""

    chunks_processed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_patterns_found: int = 0
    total_time_ms: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if self.total_input_tokens == 0:
            return 1.0
        return self.total_output_tokens / self.total_input_tokens

    @property
    def savings_percent(self) -> float:
        """Overall savings percentage."""
        return (1 - self.compression_ratio) * 100

    @property
    def throughput_tokens_per_sec(self) -> float:
        """Processing throughput."""
        if self.total_time_ms == 0:
            return 0.0
        return (self.total_input_tokens / self.total_time_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks_processed": self.chunks_processed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_patterns_found": self.total_patterns_found,
            "total_time_ms": round(self.total_time_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "compression_ratio": round(self.compression_ratio, 4),
            "savings_percent": round(self.savings_percent, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 0),
        }


@dataclass
class StreamingResult:
    """Result from compressing a single chunk."""

    chunk_index: int
    compressed_tokens: list[Token]
    dictionary_tokens: list[Token]
    body_tokens: list[Token]
    local_dictionary: dict[Token, tuple[Token, ...]]
    original_length: int
    compressed_length: int
    patterns_found: int
    time_ms: float
    is_final: bool = False
    cumulative_stats: StreamingStats | None = None
    # For overlap handling during decompression
    _overlap_tokens_at_start: int = 0  # Tokens from previous chunk's overlap

    @property
    def compression_ratio(self) -> float:
        """Chunk compression ratio."""
        if self.original_length == 0:
            return 1.0
        return self.compressed_length / self.original_length

    @property
    def savings_percent(self) -> float:
        """Chunk savings percentage."""
        return (1 - self.compression_ratio) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "chunk_index": self.chunk_index,
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "compression_ratio": round(self.compression_ratio, 4),
            "savings_percent": round(self.savings_percent, 2),
            "patterns_found": self.patterns_found,
            "time_ms": round(self.time_ms, 2),
            "is_final": self.is_final,
        }
        if self.cumulative_stats:
            result["cumulative_stats"] = self.cumulative_stats.to_dict()
        return result


def _estimate_memory_mb() -> float:
    """Estimate current memory usage in MB."""
    # Simple heuristic using sys.getsizeof for key objects
    # In production, could use tracemalloc for more accuracy
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)  # Convert to MB (macOS reports in bytes)
    except (ImportError, AttributeError):
        return 0.0


class StreamingCompressor:
    """Memory-bounded streaming compressor.

    Processes token sequences in chunks, maintaining a configurable overlap
    to detect patterns that span chunk boundaries. Each chunk is compressed
    independently with its own dictionary.

    Integrates with PatternCache for cross-document pattern learning.
    """

    def __init__(
        self,
        chunk_size: int = 8192,
        overlap_size: int = 1024,
        max_memory_mb: int = 100,
        config: CompressionConfig | None = None,
        pattern_cache: "PatternCache | None" = None,
    ) -> None:
        """Initialize streaming compressor.

        Args:
            chunk_size: Target tokens per chunk (actual may vary slightly).
            overlap_size: Tokens to overlap between chunks for cross-boundary patterns.
            max_memory_mb: Maximum memory budget in MB.
            config: Base compression config. Streaming settings can be overridden.
            pattern_cache: Optional pattern cache for cross-document learning.
        """
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if overlap_size < 0:
            raise ValueError("overlap_size cannot be negative")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if max_memory_mb < 1:
            raise ValueError("max_memory_mb must be at least 1")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_memory_mb = max_memory_mb
        self.base_config = config or CompressionConfig()
        self.pattern_cache = pattern_cache

        # Internal state
        self._buffer: list[Token] = []
        self._chunk_index = 0
        self._stats = StreamingStats()
        self._global_patterns: dict[tuple[Token, ...], Token] = {}  # pattern -> meta_token
        self._next_meta_id = 0
        self._finalized = False
        self._pending_overlap = 0  # Overlap tokens carried from previous chunk

    def add_tokens(self, tokens: Sequence[Token]) -> StreamingResult | None:
        """Add tokens to the buffer.

        When buffer reaches chunk_size, compresses and returns the chunk.
        Returns None if more tokens needed before emitting a chunk.

        Args:
            tokens: Tokens to add to the stream.

        Returns:
            StreamingResult if a chunk was emitted, None otherwise.

        Raises:
            RuntimeError: If called after flush().
        """
        if self._finalized:
            raise RuntimeError("Cannot add tokens after flush()")

        self._buffer.extend(tokens)

        # Check if we have enough for a chunk
        if len(self._buffer) >= self.chunk_size:
            return self._emit_chunk(is_final=False)

        return None

    def flush(self) -> StreamingResult:
        """Flush remaining buffer and finalize.

        Must be called after all tokens have been added. After this,
        no more tokens can be added.

        Returns:
            Final StreamingResult with is_final=True.
        """
        if self._finalized:
            raise RuntimeError("Already finalized")

        self._finalized = True
        return self._emit_chunk(is_final=True)

    def _emit_chunk(self, is_final: bool) -> StreamingResult:
        """Compress and emit a chunk."""
        start_time = time.perf_counter()

        # Track overlap from previous chunk
        overlap_at_start = self._pending_overlap

        # Determine chunk boundaries
        if is_final:
            # Process all remaining tokens
            chunk_tokens = self._buffer
            self._buffer = []
            self._pending_overlap = 0
        else:
            # Process up to chunk_size, keep overlap for next chunk
            chunk_end = self.chunk_size
            chunk_tokens = self._buffer[:chunk_end]
            # Keep overlap tokens for next chunk (these will be re-processed)
            self._buffer = self._buffer[chunk_end - self.overlap_size :]
            self._pending_overlap = self.overlap_size

        if not chunk_tokens:
            # Empty final chunk
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return StreamingResult(
                chunk_index=self._chunk_index,
                compressed_tokens=[],
                dictionary_tokens=[],
                body_tokens=[],
                local_dictionary={},
                original_length=0,
                compressed_length=0,
                patterns_found=0,
                time_ms=elapsed_ms,
                is_final=is_final,
                cumulative_stats=self._stats,
            )

        # Build chunk config with pattern cache
        chunk_config = CompressionConfig(
            min_subsequence_length=self.base_config.min_subsequence_length,
            max_subsequence_length=self.base_config.max_subsequence_length,
            selection_mode=self.base_config.selection_mode,
            discovery_mode=self.base_config.discovery_mode,
            hierarchical_enabled=self.base_config.hierarchical_enabled,
            hierarchical_max_depth=min(2, self.base_config.hierarchical_max_depth),  # Limit depth for streaming
            pattern_cache=self.pattern_cache,
            warm_start_top_k=self.base_config.warm_start_top_k,
            cache_learning_enabled=self.base_config.cache_learning_enabled,
            verify=False,  # Skip verification for performance
        )

        # Compress the chunk
        result = compress(chunk_tokens, chunk_config)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update stats
        self._stats.chunks_processed += 1
        self._stats.total_input_tokens += result.original_length
        self._stats.total_output_tokens += result.compressed_length
        self._stats.total_patterns_found += len(result.dictionary_map)
        self._stats.total_time_ms += elapsed_ms
        self._stats.peak_memory_mb = max(self._stats.peak_memory_mb, _estimate_memory_mb())

        # Track patterns globally for potential deduplication
        for meta_token, pattern in result.dictionary_map.items():
            if pattern not in self._global_patterns:
                self._global_patterns[pattern] = meta_token

        chunk_result = StreamingResult(
            chunk_index=self._chunk_index,
            compressed_tokens=result.serialized_tokens,
            dictionary_tokens=result.dictionary_tokens,
            body_tokens=result.body_tokens,
            local_dictionary=result.dictionary_map,
            original_length=result.original_length,
            compressed_length=result.compressed_length,
            patterns_found=len(result.dictionary_map),
            time_ms=elapsed_ms,
            is_final=is_final,
            cumulative_stats=StreamingStats(
                chunks_processed=self._stats.chunks_processed,
                total_input_tokens=self._stats.total_input_tokens,
                total_output_tokens=self._stats.total_output_tokens,
                total_patterns_found=self._stats.total_patterns_found,
                total_time_ms=self._stats.total_time_ms,
                peak_memory_mb=self._stats.peak_memory_mb,
            ),
            _overlap_tokens_at_start=overlap_at_start,
        )

        self._chunk_index += 1
        return chunk_result

    def compress_all(
        self, tokens: Sequence[Token], batch_size: int = 4096
    ) -> Iterator[StreamingResult]:
        """Compress all tokens, yielding results as chunks complete.

        Convenience method that feeds tokens in batches and yields
        compressed chunks as they become available.

        Args:
            tokens: Full token sequence to compress.
            batch_size: How many tokens to add per iteration.

        Yields:
            StreamingResult for each completed chunk.
        """
        n = len(tokens)
        for i in range(0, n, batch_size):
            batch = tokens[i : i + batch_size]
            result = self.add_tokens(batch)
            if result is not None:
                yield result

        # Flush final chunk
        yield self.flush()

    def get_stats(self) -> StreamingStats:
        """Get current cumulative statistics."""
        return StreamingStats(
            chunks_processed=self._stats.chunks_processed,
            total_input_tokens=self._stats.total_input_tokens,
            total_output_tokens=self._stats.total_output_tokens,
            total_patterns_found=self._stats.total_patterns_found,
            total_time_ms=self._stats.total_time_ms,
            peak_memory_mb=self._stats.peak_memory_mb,
        )

    def reset(self) -> StreamingStats:
        """Reset compressor state for reuse. Returns previous stats."""
        old_stats = self.get_stats()
        self._buffer.clear()
        self._chunk_index = 0
        self._stats = StreamingStats()
        self._global_patterns.clear()
        self._next_meta_id = 0
        self._finalized = False
        self._pending_overlap = 0
        return old_stats

    @property
    def buffer_size(self) -> int:
        """Current number of tokens in buffer."""
        return len(self._buffer)

    @property
    def is_finalized(self) -> bool:
        """Whether flush() has been called."""
        return self._finalized


def compress_streaming(
    tokens: Sequence[Token],
    chunk_size: int = 8192,
    overlap_size: int = 1024,
    config: CompressionConfig | None = None,
) -> tuple[list[list[Token]], StreamingStats]:
    """Convenience function for streaming compression.

    Compresses tokens in chunks and returns all compressed chunks
    plus cumulative statistics.

    Args:
        tokens: Token sequence to compress.
        chunk_size: Tokens per chunk.
        overlap_size: Overlap between chunks.
        config: Optional compression config.

    Returns:
        Tuple of (list of compressed token sequences, stats).

    Example:
        >>> chunks, stats = compress_streaming(tokens, chunk_size=4096)
        >>> print(f"Compressed {stats.total_input_tokens} tokens in {len(chunks)} chunks")
    """
    compressor = StreamingCompressor(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        config=config,
        pattern_cache=config.pattern_cache if config else None,
    )

    compressed_chunks: list[list[Token]] = []

    for result in compressor.compress_all(tokens):
        compressed_chunks.append(result.compressed_tokens)

    return compressed_chunks, compressor.get_stats()


def decompress_streaming(
    chunks: Sequence[Sequence[Token]] | Sequence[StreamingResult],
    config: CompressionConfig | None = None,
) -> list[Token]:
    """Decompress streaming-compressed chunks.

    Each chunk is decompressed independently and concatenated,
    with overlap regions removed to avoid duplicate tokens.

    Args:
        chunks: List of compressed chunk token sequences or StreamingResult objects.
        config: Optional compression config.

    Returns:
        Reconstructed original tokens.
    """
    from .compressor import decompress

    result: list[Token] = []
    for i, chunk in enumerate(chunks):
        # Handle both raw token sequences and StreamingResult objects
        if isinstance(chunk, StreamingResult):
            tokens = chunk.compressed_tokens
            overlap = chunk._overlap_tokens_at_start
        else:
            tokens = list(chunk)
            overlap = 0

        if not tokens:
            continue

        decompressed = decompress(tokens, config)

        # Skip overlap tokens from previous chunk (except for first chunk)
        if i > 0 and overlap > 0:
            decompressed = decompressed[overlap:]

        result.extend(decompressed)

    return result
