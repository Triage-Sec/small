"""Phase 3: dictionary construction."""

from __future__ import annotations

from typing import Iterable

from .config import CompressionConfig
from .types import Token, TokenSeq


def build_dictionary_tokens(dictionary_map: dict[Token, tuple[Token, ...]], config: CompressionConfig) -> list[Token]:
    tokens: list[Token] = [config.dict_start_token]
    for meta_token, subseq in dictionary_map.items():
        tokens.append(meta_token)
        tokens.extend(subseq)
    tokens.append(config.dict_end_token)
    return tokens


def build_body_tokens(tokens: TokenSeq, replacements: dict[int, tuple[int, Token]]) -> list[Token]:
    body: list[Token] = []
    idx = 0
    n = len(tokens)
    while idx < n:
        replacement = replacements.get(idx)
        if replacement is None:
            body.append(tokens[idx])
            idx += 1
            continue
        length, meta_token = replacement
        body.append(meta_token)
        idx += length
    return body
