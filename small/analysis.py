"""Offline analysis utilities for corpus scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingProvider


@dataclass(frozen=True)
class AnalysisConfig:
    clusters: int = 32


def compute_document_weights(
    texts: list[str], provider: EmbeddingProvider, config: AnalysisConfig
) -> list[float]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("analysis requires scikit-learn") from exc

    embeddings = np.asarray(provider.embed_batch(texts), dtype=np.float32)
    if embeddings.size == 0:
        return []
    if provider.normalize():
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    clusters = min(config.clusters, len(texts))
    kmeans = KMeans(n_clusters=clusters, n_init="auto", random_state=7)
    labels = kmeans.fit_predict(embeddings)

    counts = {label: 0 for label in range(clusters)}
    for label in labels:
        counts[label] += 1

    weights = []
    for label in labels:
        weights.append(1.0 / counts[label])
    return weights
