from small import CompressionConfig
from small.discovery import discover_candidates
from small.selection import select_occurrences


def test_selection_optimal_beats_greedy():
    tokens = ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
    cfg_greedy = CompressionConfig(selection_mode="greedy", static_dictionary_auto=False)
    cfg_opt = CompressionConfig(selection_mode="optimal", static_dictionary_auto=False)
    candidates = discover_candidates(tokens, cfg_greedy.max_subsequence_length, cfg_greedy)

    greedy = select_occurrences(candidates, cfg_greedy)
    optimal = select_occurrences(candidates, cfg_opt)

    greedy_savings = sum(occ.length - 1 for occ in greedy.selected)
    optimal_savings = sum(occ.length - 1 for occ in optimal.selected)
    assert optimal_savings >= greedy_savings
