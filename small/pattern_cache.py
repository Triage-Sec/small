"""Cross-document pattern cache for improved compression ratio.

This module provides a cache that learns patterns across multiple compression
operations, enabling "warm start" compression with pre-discovered patterns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _pattern_hash(tokens: Sequence[Any]) -> str:
    """Compute a stable hash for a token sequence."""
    # Use MD5 for speed - not security-sensitive
    content = json.dumps(list(tokens), separators=(",", ":"), sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class PatternEntry:
    """A cached pattern with usage statistics."""

    tokens: tuple[Any, ...]
    frequency: int = 1  # Times this pattern was selected for compression
    total_occurrences: int = 0  # Sum of occurrences across all compressions
    total_savings: int = 0  # Cumulative tokens saved by this pattern
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    operation_count: int = 1  # Number of compression ops that used this pattern

    @property
    def avg_occurrences(self) -> float:
        """Average occurrences per compression operation."""
        return self.total_occurrences / self.operation_count if self.operation_count else 0

    @property
    def avg_savings(self) -> float:
        """Average tokens saved per occurrence."""
        return self.total_savings / self.total_occurrences if self.total_occurrences else 0

    @property
    def length(self) -> int:
        """Pattern length in tokens."""
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tokens": list(self.tokens),
            "frequency": self.frequency,
            "total_occurrences": self.total_occurrences,
            "total_savings": self.total_savings,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "operation_count": self.operation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternEntry:
        """Deserialize from dictionary."""
        return cls(
            tokens=tuple(data["tokens"]),
            frequency=data["frequency"],
            total_occurrences=data["total_occurrences"],
            total_savings=data["total_savings"],
            first_seen=data["first_seen"],
            last_seen=data["last_seen"],
            operation_count=data["operation_count"],
        )


class BloomFilter:
    """Simple bloom filter for fast pattern membership test.

    Uses multiple hash functions to check if a pattern might be in the cache.
    False positives are possible, but false negatives are not.
    """

    def __init__(self, expected_items: int = 10000, fp_rate: float = 0.01) -> None:
        """Initialize bloom filter.

        Args:
            expected_items: Expected number of items to store.
            fp_rate: Target false positive rate (0.01 = 1%).
        """
        # Calculate optimal size and hash count
        # m = -n*ln(p) / (ln(2)^2)
        # k = (m/n) * ln(2)
        self.size = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.hash_count = max(1, int((self.size / expected_items) * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)
        self._count = 0

    def _hashes(self, item: str) -> list[int]:
        """Generate hash positions for an item."""
        # Use double hashing: h(i) = h1 + i*h2
        h1 = int(hashlib.md5(item.encode()).hexdigest()[:8], 16)
        h2 = int(hashlib.md5((item + "salt").encode()).hexdigest()[:8], 16)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << (pos % 8)
        self._count += 1

    def might_contain(self, item: str) -> bool:
        """Check if item might be in the filter.

        Returns True if the item might be present (could be false positive).
        Returns False if the item is definitely not present.
        """
        for pos in self._hashes(item):
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def clear(self) -> None:
        """Clear the filter."""
        self.bits = bytearray((self.size + 7) // 8)
        self._count = 0

    @property
    def count(self) -> int:
        """Approximate number of items added."""
        return self._count


class PatternCache:
    """Thread-safe cache for cross-document pattern learning.

    Tracks patterns that have been useful across multiple compression operations,
    enabling warm-start compression with pre-discovered high-value patterns.
    """

    def __init__(
        self,
        max_patterns: int = 10000,
        min_frequency: int = 2,
        decay_half_life: int = 100,
        min_pattern_length: int = 2,
        max_pattern_length: int = 64,
    ) -> None:
        """Initialize the pattern cache.

        Args:
            max_patterns: Maximum patterns to store before pruning.
            min_frequency: Minimum times a pattern must be selected before caching.
            decay_half_life: Number of operations after which pattern score halves.
            min_pattern_length: Minimum pattern length to cache.
            max_pattern_length: Maximum pattern length to cache.
        """
        self._lock = threading.RLock()
        self._patterns: dict[str, PatternEntry] = {}
        self._bloom = BloomFilter(expected_items=max_patterns)
        self._operation_counter = 0

        self.max_patterns = max_patterns
        self.min_frequency = min_frequency
        self.decay_half_life = decay_half_life
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length

        # Stats
        self._hits = 0
        self._misses = 0
        self._warm_starts = 0

    def record_patterns(
        self,
        dictionary_map: dict[Any, tuple[Any, ...]],
        candidate_positions: dict[tuple[Any, ...], tuple[int, ...]] | None = None,
    ) -> None:
        """Record patterns used in a compression operation.

        Args:
            dictionary_map: The dictionary from compression result (meta_token -> token sequence).
            candidate_positions: Optional mapping of patterns to their positions (for occurrence count).
        """
        with self._lock:
            self._operation_counter += 1

            for meta_token, tokens in dictionary_map.items():
                if not (self.min_pattern_length <= len(tokens) <= self.max_pattern_length):
                    continue

                token_tuple = tuple(tokens)
                pattern_hash = _pattern_hash(token_tuple)

                # Calculate savings: each occurrence saves (length - 1) tokens
                # (pattern replaced by single meta-token)
                occurrences = 1
                if candidate_positions and token_tuple in candidate_positions:
                    occurrences = len(candidate_positions[token_tuple])
                savings = occurrences * (len(tokens) - 1)

                if pattern_hash in self._patterns:
                    # Update existing entry
                    entry = self._patterns[pattern_hash]
                    entry.frequency += 1
                    entry.total_occurrences += occurrences
                    entry.total_savings += savings
                    entry.last_seen = time.time()
                    entry.operation_count += 1
                else:
                    # New pattern - only add if we have room or it's valuable
                    entry = PatternEntry(
                        tokens=token_tuple,
                        frequency=1,
                        total_occurrences=occurrences,
                        total_savings=savings,
                    )
                    self._patterns[pattern_hash] = entry
                    self._bloom.add(pattern_hash)

            # Prune if over capacity
            if len(self._patterns) > self.max_patterns:
                self._prune()

    def get_warm_start_candidates(
        self,
        tokens: Sequence[Any],
        top_k: int = 50,
    ) -> list[tuple[tuple[Any, ...], tuple[int, ...], int]]:
        """Get cached patterns that appear in the input tokens.

        Returns patterns as (subsequence, positions, priority) tuples suitable
        for creating Candidate objects.

        Args:
            tokens: Input token sequence to search.
            top_k: Maximum number of patterns to return.

        Returns:
            List of (pattern_tokens, positions, priority) tuples.
        """
        with self._lock:
            self._warm_starts += 1

            if not self._patterns:
                return []

            # Score and sort patterns by value
            scored_patterns: list[tuple[float, PatternEntry]] = []
            for entry in self._patterns.values():
                if entry.frequency < self.min_frequency:
                    continue
                score = self._score_pattern(entry)
                scored_patterns.append((score, entry))

            # Sort by score descending
            scored_patterns.sort(key=lambda x: -x[0])

            # Find patterns that appear in input
            results: list[tuple[tuple[Any, ...], tuple[int, ...], int]] = []
            token_list = list(tokens)
            n = len(token_list)

            for score, entry in scored_patterns[:top_k * 2]:  # Check more than needed
                if len(results) >= top_k:
                    break

                pattern = entry.tokens
                pattern_len = len(pattern)

                # Find all occurrences of this pattern
                positions: list[int] = []
                for i in range(n - pattern_len + 1):
                    if tuple(token_list[i : i + pattern_len]) == pattern:
                        positions.append(i)

                if len(positions) >= 2:  # Only if pattern appears multiple times
                    self._hits += 1
                    # Priority based on score (higher = better)
                    priority = int(score * 100)
                    results.append((pattern, tuple(positions), priority))
                else:
                    self._misses += 1

            return results

    def _score_pattern(self, entry: PatternEntry) -> float:
        """Calculate pattern value score.

        Score combines:
        - Frequency (how often pattern was selected)
        - Recency (when pattern was last seen)
        - Savings efficiency (tokens saved per occurrence)
        """
        # Frequency component
        freq_score = math.log1p(entry.frequency)

        # Recency component (exponential decay)
        age_ops = self._operation_counter - entry.operation_count
        recency_score = math.pow(0.5, age_ops / max(1, self.decay_half_life))

        # Savings efficiency: avg tokens saved per occurrence
        savings_score = entry.avg_savings if entry.avg_savings > 0 else 1.0

        # Length bonus: longer patterns are more valuable
        length_score = math.log1p(entry.length)

        return freq_score * recency_score * savings_score * length_score

    def _prune(self) -> None:
        """Remove low-value patterns to stay under capacity."""
        if len(self._patterns) <= self.max_patterns:
            return

        # Score all patterns
        scored: list[tuple[float, str]] = [
            (self._score_pattern(entry), hash_key)
            for hash_key, entry in self._patterns.items()
        ]
        scored.sort(key=lambda x: x[0])

        # Remove bottom 20%
        remove_count = len(self._patterns) - int(self.max_patterns * 0.8)
        for _, hash_key in scored[:remove_count]:
            del self._patterns[hash_key]

        # Rebuild bloom filter
        self._bloom.clear()
        for hash_key in self._patterns:
            self._bloom.add(hash_key)

        logger.debug(f"Pruned pattern cache: removed {remove_count} patterns")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_freq = sum(e.frequency for e in self._patterns.values())
            total_savings = sum(e.total_savings for e in self._patterns.values())

            return {
                "pattern_count": len(self._patterns),
                "max_patterns": self.max_patterns,
                "operation_count": self._operation_counter,
                "total_pattern_frequency": total_freq,
                "total_tokens_saved": total_savings,
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "warm_start_calls": self._warm_starts,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            }

    def clear(self) -> dict[str, Any]:
        """Clear the cache. Returns stats before clearing."""
        with self._lock:
            stats = self.stats()
            self._patterns.clear()
            self._bloom.clear()
            self._operation_counter = 0
            self._hits = 0
            self._misses = 0
            self._warm_starts = 0
            return stats

    def save(self, path: Path | str) -> None:
        """Save cache to disk."""
        path = Path(path)
        with self._lock:
            data = {
                "version": 1,
                "max_patterns": self.max_patterns,
                "min_frequency": self.min_frequency,
                "decay_half_life": self.decay_half_life,
                "operation_counter": self._operation_counter,
                "patterns": {k: v.to_dict() for k, v in self._patterns.items()},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        logger.info(f"Saved pattern cache to {path} ({len(self._patterns)} patterns)")

    def load(self, path: Path | str) -> bool:
        """Load cache from disk. Returns True if successful."""
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            if data.get("version") != 1:
                logger.warning(f"Unknown cache version: {data.get('version')}")
                return False

            with self._lock:
                self._operation_counter = data.get("operation_counter", 0)
                self._patterns = {
                    k: PatternEntry.from_dict(v) for k, v in data.get("patterns", {}).items()
                }
                # Rebuild bloom filter
                self._bloom.clear()
                for hash_key in self._patterns:
                    self._bloom.add(hash_key)

            logger.info(f"Loaded pattern cache from {path} ({len(self._patterns)} patterns)")
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load pattern cache: {e}")
            return False

    def __len__(self) -> int:
        """Number of patterns in cache."""
        with self._lock:
            return len(self._patterns)

    def __contains__(self, pattern: Sequence[Any]) -> bool:
        """Check if pattern is cached (may have false positives from bloom filter)."""
        pattern_hash = _pattern_hash(pattern)
        if not self._bloom.might_contain(pattern_hash):
            return False
        with self._lock:
            return pattern_hash in self._patterns
