"""Corpus preprocessing pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator

from .corpus import CorpusDocument
from .tokenizer import TokenizerAdapter
from .types import Token


@dataclass(frozen=True)
class PreprocessConfig:
    min_tokens: int = 4
    max_tokens: int = 8192
    deduplicate: bool = True


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preprocess_corpus(
    docs: Iterable[CorpusDocument],
    tokenizer: TokenizerAdapter,
    config: PreprocessConfig,
) -> Iterator[tuple[CorpusDocument, list[Token]]]:
    seen: set[str] = set()
    for doc in docs:
        if config.deduplicate:
            digest = _hash_text(doc.text)
            if digest in seen:
                continue
            seen.add(digest)

        tokens = tokenizer.encode(doc.text)
        if len(tokens) < config.min_tokens:
            continue
        if config.max_tokens and len(tokens) > config.max_tokens:
            continue
        yield doc, tokens
