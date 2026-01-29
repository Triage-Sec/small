"""Corpus loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    text: str
    domain: str
    source: str | None = None
    language: str | None = None
    created_at: str | None = None
    tags: tuple[str, ...] = ()
    token_count: int | None = None
    parent_id: str | None = None
    path: str | None = None


def load_jsonl(path: str | Path) -> Iterator[CorpusDocument]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield CorpusDocument(
                id=obj["id"],
                text=obj["text"],
                domain=obj["domain"],
                source=obj.get("source"),
                language=obj.get("language"),
                created_at=obj.get("created_at"),
                tags=tuple(obj.get("tags", [])),
                token_count=obj.get("token_count"),
                parent_id=obj.get("parent_id"),
                path=obj.get("path"),
            )


def load_directory(path: str | Path) -> Iterator[CorpusDocument]:
    path = Path(path)
    manifest = path / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for domain_info in data.get("domains", {}).values():
            if isinstance(domain_info, dict):
                for sub in domain_info.values():
                    if isinstance(sub, dict) and "path" in sub:
                        yield from load_jsonl(path / sub["path"])
        return
    for jsonl_path in path.rglob("*.jsonl"):
        yield from load_jsonl(jsonl_path)


def load_database(connection, query: str) -> Iterator[CorpusDocument]:
    cursor = connection.execute(query)
    for row in cursor.fetchall():
        obj = json.loads(row[0]) if isinstance(row[0], str) else row
        yield CorpusDocument(
            id=obj["id"],
            text=obj["text"],
            domain=obj["domain"],
            source=obj.get("source"),
            language=obj.get("language"),
            created_at=obj.get("created_at"),
            tags=tuple(obj.get("tags", [])),
            token_count=obj.get("token_count"),
            parent_id=obj.get("parent_id"),
            path=obj.get("path"),
        )
