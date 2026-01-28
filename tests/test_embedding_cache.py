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
    cache.close()
    assert restored == vector
