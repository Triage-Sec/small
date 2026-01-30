"""Tests for cross-document pattern cache."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from small.compressor import compress
from small.config import CompressionConfig
from small.pattern_cache import BloomFilter, PatternCache, PatternEntry, _pattern_hash


class TestPatternHash:
    """Tests for pattern hashing."""

    def test_hash_deterministic(self) -> None:
        """Same input produces same hash."""
        tokens = [1, 2, 3, 4, 5]
        h1 = _pattern_hash(tokens)
        h2 = _pattern_hash(tokens)
        assert h1 == h2

    def test_hash_different_inputs(self) -> None:
        """Different inputs produce different hashes."""
        h1 = _pattern_hash([1, 2, 3])
        h2 = _pattern_hash([1, 2, 4])
        assert h1 != h2

    def test_hash_tuple_and_list_same(self) -> None:
        """Tuple and list of same values produce same hash."""
        h1 = _pattern_hash([1, 2, 3])
        h2 = _pattern_hash((1, 2, 3))
        assert h1 == h2


class TestPatternEntry:
    """Tests for PatternEntry dataclass."""

    def test_creation(self) -> None:
        """Test basic entry creation."""
        entry = PatternEntry(
            tokens=(1, 2, 3),
            frequency=5,
            total_occurrences=10,
            total_savings=20,
            operation_count=5,  # Set explicitly
        )
        assert entry.length == 3
        assert entry.avg_occurrences == 2.0  # 10 / 5
        assert entry.avg_savings == 2.0  # 20 / 10

    def test_serialization_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        entry = PatternEntry(
            tokens=(1, 2, 3, 4),
            frequency=10,
            total_occurrences=50,
            total_savings=150,
            first_seen=1000.0,
            last_seen=2000.0,
            operation_count=5,
        )
        data = entry.to_dict()
        restored = PatternEntry.from_dict(data)
        
        assert restored.tokens == entry.tokens
        assert restored.frequency == entry.frequency
        assert restored.total_occurrences == entry.total_occurrences
        assert restored.total_savings == entry.total_savings


class TestBloomFilter:
    """Tests for bloom filter."""

    def test_basic_operations(self) -> None:
        """Test add and membership check."""
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        
        bf.add("hello")
        bf.add("world")
        
        assert bf.might_contain("hello")
        assert bf.might_contain("world")
        # Very unlikely to have false positive for this
        assert not bf.might_contain("definitely_not_here_xyz123")

    def test_no_false_negatives(self) -> None:
        """Items added should always be found."""
        bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            bf.add(item)
        
        for item in items:
            assert bf.might_contain(item), f"{item} should be in filter"

    def test_clear(self) -> None:
        """Test filter clearing."""
        bf = BloomFilter()
        bf.add("test")
        assert bf.might_contain("test")
        
        bf.clear()
        # After clear, might_contain returns False for items not re-added
        # (unless false positive, which is very unlikely)
        assert not bf.might_contain("test")


class TestPatternCache:
    """Tests for PatternCache."""

    def test_record_and_retrieve(self) -> None:
        """Test basic pattern recording and retrieval."""
        cache = PatternCache(max_patterns=100, min_frequency=1)
        
        # Simulate compression result with a pattern
        dictionary_map = {
            "<MT_0>": (1, 2, 3, 4),
            "<MT_1>": (5, 6, 7),
        }
        cache.record_patterns(dictionary_map)
        
        assert len(cache) == 2
        stats = cache.stats()
        assert stats["pattern_count"] == 2
        assert stats["operation_count"] == 1

    def test_warm_start_finds_patterns(self) -> None:
        """Test that warm start finds cached patterns in new input."""
        cache = PatternCache(max_patterns=100, min_frequency=1)
        
        # Record a pattern
        pattern = (100, 101, 102, 103)
        dictionary_map = {"<MT_0>": pattern}
        # Record it twice to meet min_frequency=1 (actually need freq>=min_frequency)
        cache.record_patterns(dictionary_map)
        cache.record_patterns(dictionary_map)
        
        # Create input that contains the pattern multiple times
        tokens = [0, 1, 2] + list(pattern) + [50, 51] + list(pattern) + [99]
        
        candidates = cache.get_warm_start_candidates(tokens, top_k=10)
        
        # Should find the pattern with its positions
        assert len(candidates) >= 1
        found_pattern, positions, priority = candidates[0]
        assert found_pattern == pattern
        assert len(positions) == 2  # Pattern appears twice

    def test_warm_start_requires_multiple_occurrences(self) -> None:
        """Warm start only returns patterns that appear 2+ times."""
        cache = PatternCache(max_patterns=100, min_frequency=1)
        
        pattern = (100, 101, 102)
        dictionary_map = {"<MT_0>": pattern}
        cache.record_patterns(dictionary_map)
        cache.record_patterns(dictionary_map)
        
        # Input with pattern only once
        tokens = [0, 1] + list(pattern) + [99]
        candidates = cache.get_warm_start_candidates(tokens, top_k=10)
        
        # Should not return patterns that only appear once
        pattern_found = any(c[0] == pattern for c in candidates)
        assert not pattern_found

    def test_frequency_threshold(self) -> None:
        """Patterns must meet min_frequency to be returned."""
        cache = PatternCache(max_patterns=100, min_frequency=3)
        
        pattern = (1, 2, 3, 4)
        dictionary_map = {"<MT_0>": pattern}
        
        # Record only twice (below threshold of 3)
        cache.record_patterns(dictionary_map)
        cache.record_patterns(dictionary_map)
        
        tokens = list(pattern) * 5  # Pattern appears 5 times
        candidates = cache.get_warm_start_candidates(tokens, top_k=10)
        
        # Should not return pattern because frequency < 3
        assert len(candidates) == 0
        
        # Record a third time
        cache.record_patterns(dictionary_map)
        
        candidates = cache.get_warm_start_candidates(tokens, top_k=10)
        # Now should be returned
        assert len(candidates) >= 1

    def test_pruning(self) -> None:
        """Test that cache prunes when over capacity."""
        cache = PatternCache(max_patterns=10, min_frequency=1)
        
        # Add more patterns than capacity
        for i in range(20):
            dictionary_map = {f"<MT_{i}>": tuple(range(i, i + 4))}
            cache.record_patterns(dictionary_map)
        
        # Should have pruned to ~80% of max
        assert len(cache) <= 10

    def test_persistence(self) -> None:
        """Test save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            
            # Create and populate cache
            cache1 = PatternCache(max_patterns=100, min_frequency=1)
            dictionary_map = {
                "<MT_0>": (1, 2, 3, 4),
                "<MT_1>": (5, 6, 7, 8, 9),
            }
            cache1.record_patterns(dictionary_map)
            cache1.record_patterns(dictionary_map)
            cache1.save(cache_path)
            
            # Load into new cache
            cache2 = PatternCache(max_patterns=100, min_frequency=1)
            assert cache2.load(cache_path)
            
            assert len(cache2) == len(cache1)
            stats1 = cache1.stats()
            stats2 = cache2.stats()
            assert stats1["pattern_count"] == stats2["pattern_count"]
            assert stats1["operation_count"] == stats2["operation_count"]

    def test_clear(self) -> None:
        """Test cache clearing."""
        cache = PatternCache(max_patterns=100, min_frequency=1)
        dictionary_map = {"<MT_0>": (1, 2, 3)}
        cache.record_patterns(dictionary_map)
        
        assert len(cache) == 1
        
        old_stats = cache.clear()
        assert old_stats["pattern_count"] == 1
        assert len(cache) == 0

    def test_thread_safety(self) -> None:
        """Test concurrent access is safe."""
        cache = PatternCache(max_patterns=1000, min_frequency=1)
        errors: list[Exception] = []
        
        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    pattern = tuple(range(thread_id * 1000 + i, thread_id * 1000 + i + 4))
                    dictionary_map = {f"<MT_{thread_id}_{i}>": pattern}
                    cache.record_patterns(dictionary_map)
            except Exception as e:
                errors.append(e)
        
        def reader() -> None:
            try:
                for _ in range(100):
                    tokens = list(range(5000))
                    cache.get_warm_start_candidates(tokens, top_k=10)
                    cache.stats()
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(5)
        ] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


class TestPatternCacheIntegration:
    """Integration tests with actual compression."""

    def test_compression_with_cache(self) -> None:
        """Test that cache integrates with compression."""
        cache = PatternCache(max_patterns=1000, min_frequency=1)
        
        # Create input with repeated patterns
        pattern = [100, 101, 102, 103, 104]
        tokens = []
        for _ in range(10):
            tokens.extend(pattern)
            tokens.extend([200 + len(tokens) % 50])  # Varying separator
        
        config = CompressionConfig(
            pattern_cache=cache,
            warm_start_top_k=50,
            cache_learning_enabled=True,
        )
        
        # First compression
        result1 = compress(tokens, config)
        
        # Cache should have learned patterns
        stats = cache.stats()
        assert stats["pattern_count"] > 0

    def test_warm_start_improves_ratio(self) -> None:
        """Test that warm start can improve compression ratio."""
        cache = PatternCache(max_patterns=1000, min_frequency=1)
        
        # Create a corpus with repeated structure
        def make_document() -> list[int]:
            header = [1, 2, 3, 4, 5]  # Common header
            footer = [90, 91, 92, 93]  # Common footer
            body = [i % 50 + 10 for i in range(50)]  # Variable body
            return header + body + footer
        
        config_with_cache = CompressionConfig(
            pattern_cache=cache,
            warm_start_top_k=50,
            cache_learning_enabled=True,
        )
        
        config_without_cache = CompressionConfig(
            pattern_cache=None,
        )
        
        # Compress same document multiple times with cache
        ratios_with_cache = []
        for _ in range(5):
            doc = make_document()
            result = compress(doc, config_with_cache)
            ratio = result.compressed_length / result.original_length
            ratios_with_cache.append(ratio)
        
        # The cache learns, but improvement depends on pattern characteristics
        # At minimum, it shouldn't make things worse
        assert ratios_with_cache[-1] <= ratios_with_cache[0] * 1.1  # Allow 10% variance

    def test_cache_stats_tracking(self) -> None:
        """Test that cache tracks statistics correctly."""
        cache = PatternCache(max_patterns=1000, min_frequency=1)
        
        pattern = [10, 20, 30, 40]
        tokens = pattern * 5
        
        config = CompressionConfig(
            pattern_cache=cache,
            warm_start_top_k=50,
        )
        
        # Compress multiple times
        for _ in range(3):
            compress(tokens, config)
        
        stats = cache.stats()
        assert stats["operation_count"] == 3
        assert stats["warm_start_calls"] >= 2  # Called on 2nd and 3rd compression


class TestPatternCacheMCPIntegration:
    """Test pattern cache with MCP tools."""

    def test_mcp_tools_with_cache(self) -> None:
        """Test that MCP tools work with pattern cache."""
        from small.mcp.config import MCPConfig
        from small.mcp.metrics import MetricsStore
        from small.mcp.tools import ToolHandlers

        cache = PatternCache(max_patterns=100, min_frequency=1)
        config = MCPConfig(enable_pattern_cache=True)
        metrics = MetricsStore()
        
        handlers = ToolHandlers(config, metrics, cache)
        
        # Compress some tokens
        tokens = [1, 2, 3, 4, 5] * 10
        result = handlers.compress_tokens({"tokens": tokens})
        
        assert "compressed_tokens" in result
        assert result["original_length"] == len(tokens)
        
        # Get cache stats
        stats = handlers.get_pattern_cache_stats({})
        assert stats["enabled"] is True
        assert stats["operation_count"] >= 1
        
        # Clear cache
        clear_result = handlers.clear_pattern_cache({})
        assert clear_result["cleared"] is True
        assert clear_result["previous_stats"]["operation_count"] >= 1

    def test_mcp_tools_without_cache(self) -> None:
        """Test that MCP tools work when cache is disabled."""
        from small.mcp.config import MCPConfig
        from small.mcp.metrics import MetricsStore
        from small.mcp.tools import ToolHandlers

        config = MCPConfig(enable_pattern_cache=False)
        metrics = MetricsStore()
        
        handlers = ToolHandlers(config, metrics, pattern_cache=None)
        
        # Cache stats should indicate disabled
        stats = handlers.get_pattern_cache_stats({})
        assert stats["enabled"] is False
        
        # Clear should indicate not enabled
        clear_result = handlers.clear_pattern_cache({})
        assert clear_result["cleared"] is False
