from small import CompressionConfig
from small.discovery import discover_candidates


def test_discovery_known_patterns():
    tokens = ["a", "b"] * 4 + ["c"]
    cfg = CompressionConfig(
        max_subsequence_length=2,
        static_dictionary_auto=False,
        hierarchical_enabled=False,
        dict_length_enabled=False,
    )
    candidates = discover_candidates(tokens, cfg.max_subsequence_length, cfg)
    assert any(cand.subsequence == ("a", "b") for cand in candidates)
