"""Suffix array construction and LCP utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import Token


@dataclass(frozen=True)
class SuffixArray:
    suffix_array: list[int]
    lcp: list[int]


def _rank_tokens(tokens: Iterable[Token]) -> list[int]:
    values = list(tokens)

    def sort_key(value: Token) -> tuple[str, str]:
        return (type(value).__name__, repr(value))

    unique = sorted(set(values), key=sort_key)
    mapping = {value: idx + 1 for idx, value in enumerate(unique)}
    return [mapping[value] for value in values]


def build_suffix_array(tokens: Iterable[Token]) -> SuffixArray:
    values = list(tokens)
    n = len(values)
    if n == 0:
        return SuffixArray(suffix_array=[], lcp=[])

    rank = _rank_tokens(values)
    sa = list(range(n))
    k = 1
    tmp = [0] * n
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev = sa[i - 1]
            curr = sa[i]
            prev_key = (rank[prev], rank[prev + k] if prev + k < n else -1)
            curr_key = (rank[curr], rank[curr + k] if curr + k < n else -1)
            tmp[curr] = tmp[prev] + (1 if curr_key != prev_key else 0)
        rank = tmp[:]
        if rank[sa[-1]] == n - 1:
            break
        k *= 2

    lcp = [0] * (n - 1)
    inv = [0] * n
    for i, idx in enumerate(sa):
        inv[idx] = i
    h = 0
    for i in range(n):
        pos = inv[i]
        if pos == n - 1:
            h = 0
            continue
        j = sa[pos + 1]
        while i + h < n and j + h < n and values[i + h] == values[j + h]:
            h += 1
        lcp[pos] = h
        if h > 0:
            h -= 1

    return SuffixArray(suffix_array=sa, lcp=lcp)


def lcp_intervals(sa: SuffixArray, min_len: int) -> list[tuple[int, int, int]]:
    if not sa.lcp:
        return []
    intervals: list[tuple[int, int, int]] = []
    stack: list[tuple[int, int]] = []
    for i, lcp_value in enumerate(sa.lcp):
        start = i
        while stack and stack[-1][1] > lcp_value:
            prev_start, prev_lcp = stack.pop()
            if prev_lcp >= min_len:
                intervals.append((prev_start, i, prev_lcp))
            start = prev_start
        if not stack or stack[-1][1] < lcp_value:
            stack.append((start, lcp_value))
    while stack:
        start, lcp_value = stack.pop()
        if lcp_value >= min_len:
            intervals.append((start, len(sa.lcp), lcp_value))
    return intervals
