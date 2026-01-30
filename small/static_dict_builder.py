"""Offline static dictionary builder."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CompressionConfig
from .discovery import discover_candidates
from .static_dicts import StaticDictionary
from .types import Token
from .utils import is_compressible


@dataclass(frozen=True)
class StaticDictionaryConfig:
    max_entries: int = 200


def build_static_dictionary(
    corpus: list[list[str]],
    weights: list[float],
    config: CompressionConfig,
    static_config: StaticDictionaryConfig,
    identifier: str,
) -> StaticDictionary:
    if len(corpus) != len(weights):
        raise ValueError("Corpus and weights length mismatch.")

    extra_cost = 1 if config.dict_length_enabled else 0
    totals: dict[tuple[Token, ...], dict[str, float | int]] = {}

    for doc_tokens, weight in zip(corpus, weights):
        candidates = discover_candidates(
            doc_tokens, config.max_subsequence_length, config
        )
        for cand in candidates:
            key: tuple[Token, ...] = cand.subsequence
            entry = totals.setdefault(key, {"count": 0, "weight": 0.0})
            entry["count"] += len(cand.positions)
            entry["weight"] += weight

    scored: list[tuple[float, tuple[Token, ...]]] = []
    for subseq, stats in totals.items():
        count = int(stats["count"])
        if not is_compressible(len(subseq), count, extra_cost=extra_cost):
            continue
        savings = len(subseq) * count - (1 + len(subseq) + count + extra_cost)
        score = savings * float(stats["weight"])
        scored.append((score, subseq))

    scored.sort(key=lambda item: item[0], reverse=True)
    entries: dict[Token, tuple[Token, ...]] = {}
    for idx, (_, subseq) in enumerate(scored[: static_config.max_entries]):
        entries[f"<SD_{idx}>"] = subseq

    return StaticDictionary(identifier=identifier, entries=entries)
