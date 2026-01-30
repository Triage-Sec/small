"""Tests for semantic selection module.

Note: Tests that require actual embedding providers (OpenAI, Voyage, etc.) are
marked with pytest.mark.skip by default since they require API keys. They can
be run manually with the appropriate environment variables set.
"""

import warnings

import pytest

from small.config import CompressionConfig
from small.compressor import compress
from small.selection import select_occurrences, _select_semantic_with_fallback
from small.selection_semantic import (
    SemanticWeight,
    _extract_context,
    _compute_pairwise_similarity,
    _build_occurrences,
    select_occurrences_semantic,
)
from small.types import Candidate
from small.embeddings import (
    EmbeddingProvider,
    create_provider,
    get_provider_from_config,
    PROVIDER_ALIASES,
)


# ============================================================================
# Context Extraction Tests
# ============================================================================


class TestContextExtraction:
    """Tests for context window extraction."""

    def test_extract_context_basic(self):
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        context = _extract_context(tokens, position=3, length=2, context_window=2)
        # Should include 2 before, the pattern (d, e), and 2 after
        assert "b" in context
        assert "c" in context
        assert "d" in context
        assert "e" in context
        assert "f" in context
        assert "g" in context

    def test_extract_context_start_boundary(self):
        tokens = ["a", "b", "c", "d", "e"]
        context = _extract_context(tokens, position=0, length=2, context_window=3)
        # Should handle start boundary gracefully
        assert "a" in context
        assert "b" in context

    def test_extract_context_end_boundary(self):
        tokens = ["a", "b", "c", "d", "e"]
        context = _extract_context(tokens, position=3, length=2, context_window=3)
        # Should handle end boundary gracefully
        assert "d" in context
        assert "e" in context


# ============================================================================
# Similarity Computation Tests
# ============================================================================


class TestSimilarityComputation:
    """Tests for pairwise similarity computation."""

    def test_identical_embeddings(self):
        embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        similarity = _compute_pairwise_similarity(embeddings)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_embeddings(self):
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        similarity = _compute_pairwise_similarity(embeddings)
        assert similarity == pytest.approx(0.0)

    def test_single_embedding(self):
        embeddings = [[1.0, 0.0, 0.0]]
        similarity = _compute_pairwise_similarity(embeddings)
        # Single embedding = fully similar to itself
        assert similarity == 1.0

    def test_empty_embeddings(self):
        embeddings = []
        similarity = _compute_pairwise_similarity(embeddings)
        assert similarity == 1.0


# ============================================================================
# Semantic Weight Tests
# ============================================================================


class TestSemanticWeight:
    """Tests for SemanticWeight dataclass."""

    def test_weight_creation(self):
        weight = SemanticWeight(
            subsequence=("a", "b"),
            similarity_score=0.8,
            diversity_score=0.2,
            weight_multiplier=1.2,
        )
        assert weight.similarity_score == 0.8
        assert weight.diversity_score == 0.2
        assert weight.weight_multiplier == 1.2


# ============================================================================
# Embedding Provider Tests
# ============================================================================


class TestEmbeddingProviders:
    """Tests for embedding provider factory."""

    def test_provider_aliases(self):
        assert PROVIDER_ALIASES["openai"] == "openai"
        assert PROVIDER_ALIASES["gpt"] == "openai"
        assert PROVIDER_ALIASES["voyage"] == "voyage"
        assert PROVIDER_ALIASES["sentence-transformers"] == "huggingface"
        assert PROVIDER_ALIASES["st"] == "huggingface"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_provider("unknown_provider")

    def test_get_provider_from_config_none(self):
        provider = get_provider_from_config(None)
        assert provider is None

    def test_get_provider_from_config_invalid(self):
        # Should return None instead of raising for graceful fallback
        provider = get_provider_from_config("nonexistent")
        assert provider is None


# ============================================================================
# Fallback Behavior Tests
# ============================================================================


class TestFallbackBehavior:
    """Tests for fallback when provider unavailable."""

    def test_fallback_no_provider_configured(self):
        tokens = "the quick brown fox ".split() * 20
        config = CompressionConfig(
            selection_mode="semantic",
            # No provider configured
            hierarchical_enabled=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compress(tokens, config)

            # Should warn about fallback
            assert len(w) >= 1
            assert "semantic_embedding_provider" in str(w[0].message)

        # Should still produce valid output (via fallback)
        assert result.compressed_length > 0

    def test_fallback_invalid_provider(self):
        tokens = "the quick brown fox ".split() * 20
        config = CompressionConfig(
            selection_mode="semantic",
            semantic_embedding_provider="nonexistent_provider",
            hierarchical_enabled=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compress(tokens, config)

            # Should warn about fallback
            assert len(w) >= 1

        # Should still produce valid output
        assert result.compressed_length > 0


# ============================================================================
# Selection Integration Tests
# ============================================================================


class TestSelectionIntegration:
    """Integration tests for semantic selection."""

    def test_select_occurrences_with_semantic_mode_fallback(self):
        """Test that semantic mode falls back gracefully."""
        candidates = [
            Candidate(
                subsequence=("a", "b", "c"),
                length=3,
                positions=(0, 10, 20),
                priority=0,
            ),
        ]
        tokens = list("abc" * 10)
        config = CompressionConfig(
            selection_mode="semantic",
            # No provider - should fall back
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = select_occurrences(candidates, config, tokens=tokens)

        # Should return valid result via fallback
        assert hasattr(result, "selected")


# ============================================================================
# Provider Integration Tests (require API keys)
# ============================================================================


@pytest.mark.skip(reason="Requires OPENAI_API_KEY")
class TestOpenAIProvider:
    """Tests that require OpenAI API key."""

    def test_openai_embedding_provider(self):
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = create_provider("openai", model="text-embedding-3-small")
        embeddings = provider.embed_batch(["hello world", "foo bar"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0

    def test_semantic_selection_with_openai(self):
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        tokens = "def process(x): return x ".split() * 30
        config = CompressionConfig(
            selection_mode="semantic",
            semantic_embedding_provider="openai",
            semantic_embedding_model="text-embedding-3-small",
            hierarchical_enabled=False,
        )

        result = compress(tokens, config)
        assert result.compressed_length <= result.original_length


@pytest.mark.skip(reason="Requires VOYAGE_API_KEY")
class TestVoyageProvider:
    """Tests that require Voyage API key."""

    def test_voyage_embedding_provider(self):
        import os

        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

        provider = create_provider("voyage", model="voyage-3-lite")
        embeddings = provider.embed_batch(["hello world", "foo bar"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0


@pytest.mark.skip(reason="Requires sentence-transformers package")
class TestSentenceTransformersProvider:
    """Tests that require sentence-transformers."""

    def test_sentence_transformers_provider(self):
        try:
            provider = create_provider("sentence-transformers")
            embeddings = provider.embed_batch(["hello world", "foo bar"])
            assert len(embeddings) == 2
        except ImportError:
            pytest.skip("sentence-transformers not installed")
