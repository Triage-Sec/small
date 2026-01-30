"""Offline analysis pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .analysis import AnalysisConfig, compute_document_weights
from .corpus import CorpusDocument
from .embedding_cache import SQLiteEmbeddingCache, cache_key
from .embeddings import EmbeddingProvider
from .metrics import log_cache_stats
from .metrics_writer import write_cache_stats_jsonl, write_offline_metrics_jsonl


@dataclass(frozen=True)
class OfflinePipelineConfig:
    cache_enabled: bool = True
    cache_key_version: int = 1


def embed_with_cache(
    provider: EmbeddingProvider,
    texts: list[str],
    cache: SQLiteEmbeddingCache | None,
    config: OfflinePipelineConfig,
    dimensions: int | None = None,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        if cache and config.cache_enabled:
            key = cache_key(
                type(provider).__name__,
                provider.model_id(),
                text,
                dimensions,
                config.cache_key_version,
            )
            cached = cache.get(key)
            if cached is not None:
                vectors.append(cached)
                continue
        vec = provider.embed_single(text)
        vectors.append(vec)
        if cache and config.cache_enabled:
            cache.set(key, vec, provider.model_id())
    return vectors


def run_offline_analysis(
    docs: Iterable[CorpusDocument],
    provider: EmbeddingProvider,
    cache: SQLiteEmbeddingCache | None,
    analysis_config: AnalysisConfig,
    pipeline_config: OfflinePipelineConfig,
    cache_metrics_path: str | None = None,
    unified_metrics_path: str | None = None,
) -> list[float]:
    texts = [doc.text for doc in docs]
    if not texts:
        return []

    vectors = embed_with_cache(provider, texts, cache, pipeline_config)

    # Temporarily wrap provider with fixed embeddings
    class _Provider:
        def __init__(self, vecs):
            self._vecs = vecs
            self._idx = 0

        def embed_batch(self, batch):
            start = self._idx
            self._idx += len(batch)
            return self._vecs[start : start + len(batch)]

        def normalize(self):
            return provider.normalize()

    weights = compute_document_weights(texts, _Provider(vectors), analysis_config)  # type: ignore[arg-type]
    if cache:
        stats = cache.stats()
        log_cache_stats(stats)
        if cache_metrics_path:
            write_cache_stats_jsonl(cache_metrics_path, stats)
        if unified_metrics_path:
            write_offline_metrics_jsonl(
                unified_metrics_path,
                stats,
                extra={"provider": provider.model_id()},
            )
    return weights
