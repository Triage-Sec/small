from pathlib import Path

from small.embedding_cache import SQLiteCacheConfig, SQLiteEmbeddingCache


def test_sqlite_cache_round_trip(tmp_path: Path):
    path = tmp_path / "embeddings.db"
    cfg = SQLiteCacheConfig(path=str(path), compression="none", precision="float32")
    cache = SQLiteEmbeddingCache(cfg)
    key = "abc"
    vector = [0.1, 0.2, 0.3]
    cache.set(key, vector, model_id="test")
    restored = cache.get(key)
    stats = cache.stats()
    cache.close()
    assert all(abs(a - b) < 1e-6 for a, b in zip(restored, vector))
    assert stats["sets"] == 1
    assert stats["hits"] == 1
