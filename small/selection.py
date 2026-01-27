"""Pattern selection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Occurrence
from .utils import is_compressible


@dataclass(frozen=True)
class SelectionResult:
    selected: list[Occurrence]


def _build_occurrences(candidates: Iterable[Candidate]) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    for candidate in candidates:
        for pos in candidate.positions:
            occurrences.append(
                Occurrence(
                    start=pos,
                    length=candidate.length,
                    subsequence=candidate.subsequence,
                    priority=candidate.priority,
                    patches=candidate.patches.get(pos, ()),
                )
            )
    occurrences.sort(key=lambda occ: (occ.start + occ.length, occ.start))
    return occurrences


def _non_overlapping(occurrences: list[Occurrence]) -> list[Occurrence]:
    selected: list[Occurrence] = []
    next_free = -1
    for occ in sorted(occurrences, key=lambda occ: (-occ.priority, occ.start, occ.length)):
        if occ.start >= next_free:
            selected.append(occ)
            next_free = occ.start + occ.length
    return selected


def _group_by_subsequence(occurrences: list[Occurrence]) -> dict[tuple, list[Occurrence]]:
    grouped: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        grouped.setdefault(occ.subsequence, []).append(occ)
    return grouped


def _filter_by_compressibility(occurrences: list[Occurrence], config: CompressionConfig) -> list[Occurrence]:
    grouped = _group_by_subsequence(occurrences)
    filtered: list[Occurrence] = []
    for subseq, occs in grouped.items():
        extra_cost = 1 if config.dict_length_enabled else 0
        if is_compressible(len(subseq), len(occs), extra_cost=extra_cost):
            filtered.extend(occs)
    filtered.sort(key=lambda occ: occ.start)
    return filtered


def _weighted_interval_scheduling(occurrences: list[Occurrence]) -> list[Occurrence]:
    if not occurrences:
        return []
    # Sort by end position
    occs = sorted(occurrences, key=lambda occ: (occ.start + occ.length, occ.start))
    ends = [occ.start + occ.length for occ in occs]

    # p[i]: last index < i that doesn't overlap
    p: list[int] = []
    for i, occ in enumerate(occs):
        lo = 0
        hi = i - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ends[mid] <= occ.start:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        p.append(idx)

    # dp over occurrences
    weights = [occ.length - 1 + occ.priority for occ in occs]
    dp = [0] * len(occs)
    choose = [False] * len(occs)
    for i in range(len(occs)):
        take = weights[i] + (dp[p[i]] if p[i] >= 0 else 0)
        skip = dp[i - 1] if i > 0 else 0
        if take > skip:
            dp[i] = take
            choose[i] = True
        else:
            dp[i] = skip
            choose[i] = False

    # Reconstruct
    selected: list[Occurrence] = []
    i = len(occs) - 1
    while i >= 0:
        if choose[i]:
            selected.append(occs[i])
            i = p[i]
        else:
            i -= 1
    selected.reverse()
    return selected


def _beam_search(occurrences: list[Occurrence], width: int) -> list[Occurrence]:
    if not occurrences:
        return []
    occs = sorted(occurrences, key=lambda occ: (occ.start, occ.length))
    # Each state: (score, last_end, selected)
    states: list[tuple[int, int, list[Occurrence]]] = [(0, -1, [])]
    for occ in occs:
        new_states: list[tuple[int, int, list[Occurrence]]] = []
        for score, last_end, selected in states:
            # skip
            new_states.append((score, last_end, selected))
            # take
            if occ.start >= last_end:
                new_selected = selected + [occ]
                new_states.append((score + (occ.length - 1 + occ.priority), occ.start + occ.length, new_selected))
        # keep top-k by score, then by shortest last_end
        new_states.sort(key=lambda s: (s[0], -s[1]), reverse=True)
        states = new_states[: max(1, width)]
    states.sort(key=lambda s: s[0], reverse=True)
    return states[0][2]


def select_occurrences(candidates: Iterable[Candidate], config: CompressionConfig) -> SelectionResult:
    occurrences = _build_occurrences(candidates)
    if config.selection_mode == "greedy":
        selected = _non_overlapping(occurrences)
    elif config.selection_mode == "optimal":
        selected = _weighted_interval_scheduling(occurrences)
    elif config.selection_mode == "beam":
        selected = _beam_search(occurrences, config.beam_width)
    else:
        raise ValueError("Unsupported selection mode.")

    selected = _filter_by_compressibility(selected, config)
    return SelectionResult(selected=selected)
