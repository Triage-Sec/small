"""Vocabulary extension helpers for meta-tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig


@dataclass(frozen=True)
class VocabExtension:
    meta_tokens: list[str]
    special_tokens: list[str]


def build_meta_tokens(config: CompressionConfig) -> list[str]:
    return [
        f"{config.meta_token_prefix}{i}{config.meta_token_suffix}"
        for i in range(config.meta_token_pool_size)
    ]


def build_special_tokens(config: CompressionConfig) -> list[str]:
    tokens = [config.dict_start_token, config.dict_end_token]
    if config.dict_length_enabled:
        tokens.append(f"{config.dict_length_prefix}0{config.dict_length_suffix}")
    if config.fuzzy_enabled:
        tokens.extend([config.patch_start_token, config.patch_end_token])
        tokens.append(f"{config.patch_index_prefix}0{config.patch_index_suffix}")
    if config.static_dictionary_id or config.static_dictionary_auto:
        tokens.append(
            f"{config.static_dictionary_marker_prefix}example{config.static_dictionary_marker_suffix}"
        )
    return tokens


def plan_vocab_extension(config: CompressionConfig) -> VocabExtension:
    meta = build_meta_tokens(config)
    special = build_special_tokens(config)
    return VocabExtension(meta_tokens=meta, special_tokens=special)


def add_tokens_to_hf_tokenizer(tokenizer, tokens: Iterable[str]) -> int:
    if not hasattr(tokenizer, "add_tokens"):
        raise ValueError("Tokenizer does not support add_tokens.")
    return tokenizer.add_tokens(list(tokens))
