from pathlib import Path

from small.metrics_writer import write_offline_metrics_jsonl


def test_write_offline_metrics_jsonl(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    write_offline_metrics_jsonl(path, {"sets": 1, "hits": 2, "misses": 1, "evictions": 0}, {"provider": "x"})
    content = path.read_text(encoding="utf-8").strip()
    assert "cache_hit_rate" in content
    assert "provider" in content
