"""JSONL metrics writer."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .metrics import CompressionMetrics


def write_metrics_jsonl(
    path: str | Path, metrics: CompressionMetrics, extra: dict | None = None
) -> None:
    entry = asdict(metrics)
    entry["timestamp"] = int(time.time())
    if extra:
        entry.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def write_cache_stats_jsonl(path: str | Path, stats: dict[str, int]) -> None:
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    entry: dict[str, int | float] = {
        "timestamp": int(time.time()),
        "cache_sets": stats.get("sets", 0),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_evictions": stats.get("evictions", 0),
        "cache_hit_rate": hits / (hits + misses) if (hits + misses) else 0.0,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def write_offline_metrics_jsonl(
    path: str | Path,
    cache_stats: dict[str, int],
    extra: dict | None = None,
) -> None:
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    entry: dict[str, int | float] = {
        "timestamp": int(time.time()),
        "cache_sets": cache_stats.get("sets", 0),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_evictions": cache_stats.get("evictions", 0),
        "cache_hit_rate": hits / (hits + misses) if (hits + misses) else 0.0,
    }
    if extra:
        entry.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def write_combined_metrics_jsonl(
    path: str | Path,
    metrics: CompressionMetrics,
    cache_stats: dict[str, int],
    extra: dict | None = None,
) -> None:
    entry = asdict(metrics)
    entry["timestamp"] = int(time.time())
    entry.update(
        {
            "cache_sets": cache_stats.get("sets", 0),
            "cache_hits": cache_stats.get("hits", 0),
            "cache_misses": cache_stats.get("misses", 0),
            "cache_evictions": cache_stats.get("evictions", 0),
        }
    )
    hits = entry["cache_hits"]
    misses = entry["cache_misses"]
    entry["cache_hit_rate"] = hits / (hits + misses) if (hits + misses) else 0.0
    if extra:
        entry.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
