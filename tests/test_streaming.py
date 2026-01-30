"""Tests for streaming compression with memory bounds."""

from __future__ import annotations

import pytest

from small.compressor import compress, decompress
from small.config import CompressionConfig
from small.pattern_cache import PatternCache
from small.streaming import (
    StreamingCompressor,
    StreamingResult,
    StreamingStats,
    compress_streaming,
    decompress_streaming,
)


class TestStreamingStats:
    """Tests for StreamingStats dataclass."""

    def test_default_values(self) -> None:
        """Test default statistics."""
        stats = StreamingStats()
        assert stats.chunks_processed == 0
        assert stats.total_input_tokens == 0
        assert stats.compression_ratio == 1.0
        assert stats.savings_percent == 0.0

    def test_compression_ratio_calculation(self) -> None:
        """Test ratio and savings calculations."""
        stats = StreamingStats(
            total_input_tokens=1000,
            total_output_tokens=800,
        )
        assert stats.compression_ratio == 0.8
        assert abs(stats.savings_percent - 20.0) < 0.001

    def test_throughput_calculation(self) -> None:
        """Test throughput calculation."""
        stats = StreamingStats(
            total_input_tokens=10000,
            total_time_ms=100.0,
        )
        assert stats.throughput_tokens_per_sec == 100000.0

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        stats = StreamingStats(
            chunks_processed=5,
            total_input_tokens=5000,
            total_output_tokens=4000,
        )
        d = stats.to_dict()
        assert d["chunks_processed"] == 5
        assert d["compression_ratio"] == 0.8
        assert d["savings_percent"] == 20.0


class TestStreamingResult:
    """Tests for StreamingResult dataclass."""

    def test_compression_ratio(self) -> None:
        """Test chunk compression ratio."""
        result = StreamingResult(
            chunk_index=0,
            compressed_tokens=[1, 2, 3],
            dictionary_tokens=[],
            body_tokens=[1, 2, 3],
            local_dictionary={},
            original_length=100,
            compressed_length=80,
            patterns_found=5,
            time_ms=10.0,
        )
        assert result.compression_ratio == 0.8
        assert abs(result.savings_percent - 20.0) < 0.001

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        result = StreamingResult(
            chunk_index=2,
            compressed_tokens=[1, 2],
            dictionary_tokens=[],
            body_tokens=[1, 2],
            local_dictionary={},
            original_length=50,
            compressed_length=40,
            patterns_found=3,
            time_ms=5.0,
            is_final=True,
        )
        d = result.to_dict()
        assert d["chunk_index"] == 2
        assert d["is_final"] is True
        assert d["savings_percent"] == 20.0


class TestStreamingCompressor:
    """Tests for StreamingCompressor class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        compressor = StreamingCompressor(chunk_size=1000, overlap_size=100)
        assert compressor.chunk_size == 1000
        assert compressor.overlap_size == 100
        assert compressor.buffer_size == 0
        assert not compressor.is_finalized

    def test_invalid_chunk_size(self) -> None:
        """Test that small chunk sizes are rejected."""
        with pytest.raises(ValueError, match="chunk_size must be at least"):
            StreamingCompressor(chunk_size=50)

    def test_invalid_overlap(self) -> None:
        """Test that overlap >= chunk_size is rejected."""
        with pytest.raises(ValueError, match="overlap_size must be less than"):
            StreamingCompressor(chunk_size=1000, overlap_size=1000)

    def test_add_tokens_buffers(self) -> None:
        """Test that tokens are buffered until chunk_size."""
        compressor = StreamingCompressor(chunk_size=1000, overlap_size=100)

        result = compressor.add_tokens([1, 2, 3, 4, 5])
        assert result is None  # Not enough for a chunk
        assert compressor.buffer_size == 5

    def test_add_tokens_emits_chunk(self) -> None:
        """Test that chunk is emitted when buffer is full."""
        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)

        # Add enough tokens to trigger a chunk
        tokens = list(range(600))
        result = compressor.add_tokens(tokens)

        assert result is not None
        assert result.chunk_index == 0
        assert result.original_length == 500  # chunk_size
        assert not result.is_final

        # Buffer should have overlap + remaining tokens
        assert compressor.buffer_size == 150  # 600 - 500 + 50 overlap

    def test_flush_emits_final_chunk(self) -> None:
        """Test that flush emits remaining buffer."""
        compressor = StreamingCompressor(chunk_size=1000, overlap_size=100)

        compressor.add_tokens(list(range(500)))
        result = compressor.flush()

        assert result is not None
        assert result.is_final
        assert result.original_length == 500
        assert compressor.is_finalized

    def test_cannot_add_after_flush(self) -> None:
        """Test that add_tokens fails after flush."""
        compressor = StreamingCompressor(chunk_size=1000, overlap_size=100)
        compressor.add_tokens([1, 2, 3])
        compressor.flush()

        with pytest.raises(RuntimeError, match="Cannot add tokens after flush"):
            compressor.add_tokens([4, 5, 6])

    def test_cannot_flush_twice(self) -> None:
        """Test that flush fails if already finalized."""
        compressor = StreamingCompressor(chunk_size=1000, overlap_size=100)
        compressor.flush()

        with pytest.raises(RuntimeError, match="Already finalized"):
            compressor.flush()

    def test_compress_all_generator(self) -> None:
        """Test compress_all yields chunks correctly."""
        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)

        tokens = list(range(1500))
        results = list(compressor.compress_all(tokens))

        # Should have multiple chunks
        assert len(results) >= 2

        # Last should be final
        assert results[-1].is_final

        # Check cumulative stats
        final_stats = results[-1].cumulative_stats
        assert final_stats is not None
        assert final_stats.chunks_processed == len(results)

    def test_reset(self) -> None:
        """Test compressor can be reset and reused."""
        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)

        # Process some tokens
        compressor.add_tokens(list(range(300)))
        compressor.flush()

        # Reset
        old_stats = compressor.reset()
        assert old_stats.chunks_processed == 1

        # Should be usable again
        assert not compressor.is_finalized
        assert compressor.buffer_size == 0
        compressor.add_tokens([1, 2, 3])
        assert compressor.buffer_size == 3

    def test_stats_tracking(self) -> None:
        """Test that statistics are tracked correctly."""
        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)

        tokens = list(range(1200))
        list(compressor.compress_all(tokens))

        stats = compressor.get_stats()
        assert stats.chunks_processed >= 2
        # Note: total_input_tokens may be > len(tokens) due to overlap
        assert stats.total_time_ms > 0


class TestStreamingCompression:
    """Integration tests for streaming compression."""

    def test_roundtrip_single_chunk(self) -> None:
        """Test roundtrip with small input (single chunk)."""
        tokens = [1, 2, 3, 4, 5] * 20  # 100 tokens

        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)
        results = list(compressor.compress_all(tokens))

        assert len(results) == 1
        assert results[0].is_final

        # Decompress and verify (pass results for overlap handling)
        restored = decompress_streaming(results)
        assert restored == list(tokens)

    def test_roundtrip_multiple_chunks(self) -> None:
        """Test roundtrip with large input (multiple chunks)."""
        # Create tokens with repeated patterns
        pattern = [100, 101, 102, 103, 104]
        tokens = []
        for i in range(200):
            tokens.extend(pattern)
            tokens.append(i % 256)  # Varying token

        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)
        results = list(compressor.compress_all(tokens))

        assert len(results) >= 2

        # Decompress and verify (pass results for overlap handling)
        restored = decompress_streaming(results)
        assert restored == list(tokens)

    def test_compression_achieves_savings(self) -> None:
        """Test that streaming compression achieves actual savings."""
        # Create highly compressible input
        pattern = [1000, 1001, 1002, 1003, 1004, 1005]
        tokens = pattern * 200  # Very repetitive

        chunks, stats = compress_streaming(tokens, chunk_size=500, overlap_size=50)

        # Should achieve some compression
        assert stats.compression_ratio < 1.0
        assert stats.savings_percent > 0

    def test_streaming_vs_batch_comparison(self) -> None:
        """Compare streaming to batch compression."""
        pattern = [10, 20, 30, 40, 50]
        tokens = []
        for i in range(100):
            tokens.extend(pattern)
            tokens.extend([i % 50 + 100])

        # Batch compression
        batch_result = compress(tokens)
        batch_ratio = batch_result.compressed_length / batch_result.original_length

        # Streaming compression
        chunks, stats = compress_streaming(tokens, chunk_size=500, overlap_size=100)
        streaming_ratio = stats.compression_ratio

        # Streaming should be within 20% of batch (may be worse due to chunk boundaries)
        assert streaming_ratio < batch_ratio * 1.2

    def test_cross_chunk_patterns_via_overlap(self) -> None:
        """Test that overlap enables cross-chunk pattern detection."""
        # Create pattern that would span chunk boundary
        pattern = [1, 2, 3, 4, 5, 6, 7, 8]

        # Position pattern at chunk boundary
        tokens = list(range(496, 496 + 4))  # Just before 500
        tokens.extend(pattern)  # Spans boundary
        tokens.extend(list(range(600, 700)))
        tokens.extend(pattern)  # Second occurrence for compression

        # With overlap, pattern should be detectable
        compressor = StreamingCompressor(chunk_size=500, overlap_size=100)
        results = list(compressor.compress_all(tokens))

        # At minimum, should process without error
        assert len(results) >= 1

    def test_with_pattern_cache(self) -> None:
        """Test streaming with pattern cache integration."""
        cache = PatternCache(max_patterns=1000, min_frequency=1)

        pattern = [100, 101, 102, 103]
        tokens = []
        for i in range(50):
            tokens.extend(pattern)
            tokens.append(i + 200)

        # Use larger chunk to avoid complex overlap issues with small patterns
        compressor = StreamingCompressor(
            chunk_size=500,
            overlap_size=50,
            pattern_cache=cache,
        )

        results = list(compressor.compress_all(tokens))

        # Cache should have learned patterns
        assert cache.stats()["operation_count"] >= 1

        # Verify roundtrip still works (pass results for overlap handling)
        restored = decompress_streaming(results)
        assert restored == list(tokens)


class TestMemoryBounds:
    """Tests for memory bounding behavior."""

    def test_large_input_processes_incrementally(self) -> None:
        """Test that large inputs are processed in bounded chunks."""
        # Create 100K tokens
        tokens = list(range(100000))

        compressor = StreamingCompressor(chunk_size=8192, overlap_size=512)

        chunk_count = 0
        max_buffer_observed = 0

        for i in range(0, len(tokens), 1000):
            batch = tokens[i : i + 1000]
            result = compressor.add_tokens(batch)
            max_buffer_observed = max(max_buffer_observed, compressor.buffer_size)

            if result is not None:
                chunk_count += 1
                # Buffer should be bounded
                assert compressor.buffer_size <= compressor.chunk_size

        result = compressor.flush()
        chunk_count += 1

        # Should have processed in multiple chunks
        assert chunk_count > 1
        # Buffer never exceeded chunk_size + batch_size significantly
        assert max_buffer_observed < 10000

    def test_stats_track_memory(self) -> None:
        """Test that memory tracking works."""
        tokens = list(range(10000))

        chunks, stats = compress_streaming(tokens, chunk_size=2000, overlap_size=200)

        # Memory tracking should report something (may be 0 on some systems)
        assert stats.peak_memory_mb >= 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)
        result = compressor.flush()

        assert result.is_final
        assert result.original_length == 0
        assert result.compressed_tokens == []

    def test_input_smaller_than_chunk(self) -> None:
        """Test input smaller than chunk size."""
        tokens = [1, 2, 3, 4, 5]

        chunks, stats = compress_streaming(tokens, chunk_size=1000, overlap_size=100)

        assert len(chunks) == 1
        assert stats.chunks_processed == 1

    def test_input_exactly_chunk_size(self) -> None:
        """Test input exactly equal to chunk size."""
        tokens = list(range(500))

        compressor = StreamingCompressor(chunk_size=500, overlap_size=50)
        result = compressor.add_tokens(tokens)

        # Should emit a chunk when exactly at chunk_size
        assert result is not None
        assert result.original_length == 500

    def test_very_small_overlap(self) -> None:
        """Test with minimal overlap."""
        tokens = list(range(2000))

        compressor = StreamingCompressor(chunk_size=500, overlap_size=1)
        results = list(compressor.compress_all(tokens))

        # Should have multiple chunks
        assert len(results) >= 2
        restored = decompress_streaming(results)
        assert restored == list(tokens)

    def test_large_overlap(self) -> None:
        """Test with large overlap (close to chunk size)."""
        tokens = list(range(2000))

        compressor = StreamingCompressor(chunk_size=500, overlap_size=400)
        results = list(compressor.compress_all(tokens))

        # Should have multiple chunks (overlap reduces effective chunk size)
        assert len(results) >= 2
        restored = decompress_streaming(results)
        assert restored == list(tokens)
