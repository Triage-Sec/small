"""Fuzzy pattern discovery with minimal patch encoding."""

from __future__ import annotations

from .config import CompressionConfig
from .types import Candidate, Patch, Token, TokenSeq
from .utils import is_meta_token


def _signature(subseq: tuple[Token, ...]) -> tuple:
    return (len(subseq), tuple(subseq[::2]))


def _hamming(a: tuple[Token, ...], b: tuple[Token, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _patches(
    canonical: tuple[Token, ...], variant: tuple[Token, ...]
) -> tuple[Patch, ...]:
    diffs: list[Patch] = []
    for idx, (left, right) in enumerate(zip(canonical, variant)):
        if left != right:
            diffs.append((idx, right))
    return tuple(diffs)


def _compression_savings(
    length: int,
    patches: list[tuple[Patch, ...]],
    config: CompressionConfig,
) -> int:
    original = length * len(patches)
    dict_cost = 1 + length + (1 if config.dict_length_enabled else 0)
    body_cost = 0
    for patch in patches:
        if not patch:
            body_cost += 1
        else:
            body_cost += 1 + 2 + len(patch) * 2
    return original - (dict_cost + body_cost)


def discover_fuzzy_candidates(
    tokens: TokenSeq, config: CompressionConfig
) -> list[Candidate]:
    if not tokens:
        return []
    if not all(isinstance(tok, str) for tok in tokens):
        raise ValueError("Fuzzy matching requires string tokens.")

    candidates: list[Candidate] = []
    n = len(tokens)
    max_len = config.max_subsequence_length
    min_len = config.min_subsequence_length

    for length in range(max_len, min_len - 1, -1):
        if length > n:
            continue
        positions_by_subseq: dict[tuple[Token, ...], list[int]] = {}
        for idx in range(0, n - length + 1):
            subseq = tuple(tokens[idx : idx + length])
            if any(is_meta_token(tok, config) for tok in subseq):
                continue
            positions_by_subseq.setdefault(subseq, []).append(idx)

        buckets: dict[tuple, list[tuple[Token, ...]]] = {}
        for subseq in positions_by_subseq:
            buckets.setdefault(_signature(subseq), []).append(subseq)

        seen: set[tuple[Token, ...]] = set()
        for bucket in buckets.values():
            for subseq in bucket:
                if subseq in seen:
                    continue
                cluster = [subseq]
                seen.add(subseq)
                for other in bucket:
                    if other in seen:
                        continue
                    if _hamming(subseq, other) <= config.fuzzy_max_diff:
                        cluster.append(other)
                        seen.add(other)

                if len(cluster) < 2:
                    continue

                canonical = max(cluster, key=lambda s: len(positions_by_subseq[s]))
                patches: list[tuple[Patch, ...]] = []
                patches_by_position: dict[int, tuple[Patch, ...]] = {}
                all_positions: list[int] = []
                for variant in cluster:
                    patch = _patches(canonical, variant)
                    for pos in positions_by_subseq[variant]:
                        all_positions.append(pos)
                        patches.append(patch)
                        if patch:
                            patches_by_position[pos] = patch

                if _compression_savings(len(canonical), patches, config) <= 0:
                    continue

                all_positions.sort()
                candidates.append(
                    Candidate(
                        subsequence=canonical,
                        length=len(canonical),
                        positions=tuple(all_positions),
                        priority=config.fuzzy_priority_bonus,
                        patches=patches_by_position,
                    )
                )

    return candidates
