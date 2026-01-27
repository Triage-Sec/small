import json
from pathlib import Path

from small import CompressionConfig, compress


def test_regression_corpus():
    fixture = Path(__file__).parent / "fixtures" / "regression.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    cfg = CompressionConfig(
        static_dictionary_auto=False,
        hierarchical_enabled=False,
        dict_length_enabled=False,
        selection_mode="optimal",
    )
    for case in data:
        result = compress(case["tokens"], cfg)
        ratio = result.compressed_length / result.original_length if result.original_length else 1.0
        assert ratio <= case["min_ratio"]
