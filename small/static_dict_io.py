"""Static dictionary serialization for offline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .static_dicts import StaticDictionary


def save_static_dictionary(path: str | Path, dictionary: StaticDictionary) -> None:
    path = Path(path)
    data = {
        "identifier": dictionary.identifier,
        "entries": {k: list(v) for k, v in dictionary.entries.items()},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_static_dictionary(path: str | Path) -> StaticDictionary:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = {k: tuple(v) for k, v in data["entries"].items()}
    return StaticDictionary(identifier=data["identifier"], entries=entries)
