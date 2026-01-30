"""Embedding cache backends."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def cache_key(
    provider: str, model_id: str, text: str, dimensions: int | None, version: int
) -> str:
    normalized = _normalize_text(text)
    raw = f"{provider}|{model_id}|{normalized}|{dimensions}|{version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compress(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    if compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("zstandard is required for zstd compression.") from exc
        return zstd.ZstdCompressor().compress(data)
    raise ValueError("Unsupported compression.")


def _decompress(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    if compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("zstandard is required for zstd compression.") from exc
        return zstd.ZstdDecompressor().decompress(data)
    raise ValueError("Unsupported compression.")


@dataclass(frozen=True)
class SQLiteCacheConfig:
    path: str
    max_size_gb: float = 10.0
    compression: str = "zstd"
    precision: str = "float32"
    wal_mode: bool = True


@dataclass(frozen=True)
class RedisCacheConfig:
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "small:emb:"
    ttl_days: int | None = None


class SQLiteEmbeddingCache:
    def __init__(self, config: SQLiteCacheConfig) -> None:
        self.config = config
        Path(self.config.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.config.path)
        self.conn.execute(
            "PRAGMA journal_mode=WAL;"
            if self.config.wal_mode
            else "PRAGMA journal_mode=DELETE;"
        )
        self._init_schema()
        self._init_meta()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_id ON embeddings(model_id);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed);"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self.conn.commit()

    def _init_meta(self) -> None:
        defaults = {
            "sets": "0",
            "hits": "0",
            "misses": "0",
            "evictions": "0",
        }
        for key, value in defaults.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO cache_meta(key, value) VALUES (?, ?);",
                (key, value),
            )
        self.conn.commit()

    def _bump_meta(self, key: str, delta: int = 1) -> None:
        row = self.conn.execute(
            "SELECT value FROM cache_meta WHERE key = ?;", (key,)
        ).fetchone()
        current = int(row[0]) if row else 0
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_meta(key, value) VALUES (?, ?);",
            (key, str(current + delta)),
        )

    def _serialize(self, vector: list[float]) -> bytes:
        dtype = np.float16 if self.config.precision == "float16" else np.float32
        array = np.asarray(vector, dtype=dtype)
        return _compress(array.tobytes(), self.config.compression)

    def _deserialize(self, blob: bytes, dimension: int) -> list[float]:
        dtype = np.float16 if self.config.precision == "float16" else np.float32
        raw = _decompress(blob, self.config.compression)
        array = np.frombuffer(raw, dtype=dtype, count=dimension)
        return array.astype(np.float32).tolist()

    def get(self, key: str) -> list[float] | None:
        row = self.conn.execute(
            "SELECT embedding, dimension FROM embeddings WHERE cache_key = ?;",
            (key,),
        ).fetchone()
        if row is None:
            self._bump_meta("misses", 1)
            self.conn.commit()
            return None
        blob, dimension = row
        self._bump_meta("hits", 1)
        self.conn.execute(
            "UPDATE embeddings SET access_count = access_count + 1, last_accessed = ? WHERE cache_key = ?;",
            (int(time.time()), key),
        )
        self.conn.commit()
        return self._deserialize(blob, int(dimension))

    def set(self, key: str, vector: list[float], model_id: str) -> None:
        now = int(time.time())
        blob = self._serialize(vector)
        dimension = len(vector)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings(cache_key, embedding, model_id, dimension, created_at, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, 1, ?);
            """,
            (key, blob, model_id, dimension, now, now),
        )
        self._bump_meta("sets", 1)
        self.conn.commit()
        self._evict_if_needed()

    def _db_size_bytes(self) -> int:
        try:
            return os.path.getsize(self.config.path)
        except OSError:
            return 0

    def _evict_if_needed(self) -> None:
        limit_bytes = int(self.config.max_size_gb * 1024 * 1024 * 1024)
        if limit_bytes <= 0:
            return
        size = self._db_size_bytes()
        if size <= limit_bytes:
            return
        # Evict least recently used 10% at a time.
        rows = self.conn.execute(
            "SELECT cache_key FROM embeddings ORDER BY last_accessed ASC LIMIT (SELECT COUNT(*) / 10 FROM embeddings);"
        ).fetchall()
        if not rows:
            return
        keys = [row[0] for row in rows]
        self.conn.executemany(
            "DELETE FROM embeddings WHERE cache_key = ?;", [(k,) for k in keys]
        )
        self._bump_meta("evictions", len(keys))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT key, value FROM cache_meta;").fetchall()
        return {key: int(value) for key, value in rows}


class RedisEmbeddingCache:
    def __init__(
        self,
        config: RedisCacheConfig,
        compression: str = "zstd",
        precision: str = "float32",
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError("Redis embedding cache requires 'redis'.") from exc
        self._redis = redis.Redis.from_url(config.url)
        self.config = config
        self.compression = compression
        self.precision = precision
        self._stats_key = f"{self.config.key_prefix}stats"

    def _serialize(self, vector: list[float]) -> bytes:
        dtype = np.float16 if self.precision == "float16" else np.float32
        array = np.asarray(vector, dtype=dtype)
        return _compress(array.tobytes(), self.compression)

    def _deserialize(self, blob: bytes, dimension: int) -> list[float]:
        dtype = np.float16 if self.precision == "float16" else np.float32
        raw = _decompress(blob, self.compression)
        array = np.frombuffer(raw, dtype=dtype, count=dimension)
        return array.astype(np.float32).tolist()

    def _key(self, key: str) -> str:
        return f"{self.config.key_prefix}{key}"

    def get(self, key: str, dimension: int) -> list[float] | None:
        data = self._redis.get(self._key(key))
        if data is None:
            self._redis.hincrby(self._stats_key, "misses", 1)
            return None
        self._redis.hincrby(self._stats_key, "hits", 1)
        # Redis sync client returns bytes, cast for type checker
        return self._deserialize(
            data if isinstance(data, bytes) else bytes(str(data), "utf-8"), dimension
        )

    def set(self, key: str, vector: list[float]) -> None:
        ttl = None
        if self.config.ttl_days is not None:
            ttl = int(self.config.ttl_days * 86400)
        self._redis.set(self._key(key), self._serialize(vector), ex=ttl)
        self._redis.hincrby(self._stats_key, "sets", 1)

    def stats(self) -> dict[str, int]:
        data = self._redis.hgetall(self._stats_key)
        if not data:
            return {"sets": 0, "hits": 0, "misses": 0}
        result: dict[str, int] = {}
        for k, v in dict(data).items():  # type: ignore[arg-type]
            result[k.decode("utf-8")] = int(v)
        return result
