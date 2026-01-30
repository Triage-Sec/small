"""Serialization helpers for compressed sequences."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CompressionConfig
from .dictionary import build_dictionary_tokens
from .dictionary_store import CompressionDictionary
from .static_dicts import static_dictionary_marker
from .types import Token


@dataclass(frozen=True)
class SerializedOutput:
    tokens: list[Token]
    dictionary_tokens: list[Token]
    body_tokens: list[Token]


def serialize(
    dictionary: CompressionDictionary,
    body_tokens: list[Token],
    config: CompressionConfig,
    static_dictionary_id: str | None = None,
) -> SerializedOutput:
    dictionary_tokens = (
        build_dictionary_tokens(dictionary.meta_to_seq, config)
        if dictionary.meta_to_seq
        else []
    )
    if static_dictionary_id:
        marker = static_dictionary_marker(static_dictionary_id, config)
        tokens = (
            [marker]
            + (dictionary_tokens or [config.dict_start_token, config.dict_end_token])
            + body_tokens
        )
    elif dictionary_tokens:
        tokens = dictionary_tokens + body_tokens
    else:
        tokens = list(body_tokens)
    return SerializedOutput(
        tokens=tokens, dictionary_tokens=dictionary_tokens, body_tokens=body_tokens
    )
